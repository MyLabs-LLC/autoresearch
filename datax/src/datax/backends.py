"""Judge backends -- two ways to get a classification out of Claude.

``anthropic``
    Direct Messages API calls. Needs ``ANTHROPIC_API_KEY`` (or an ``ant auth login``
    profile). Supports the Batches API, which halves the price and is the right choice
    for building a large corpus.

``claude-code``
    Drives the local Claude Code agent through ``claude-agent-sdk``. Needs no separate
    API key -- it reuses whatever credentials Claude Code already has -- so it works on
    a developer machine or in an agent container out of the box. No batch API.

Both produce the same :class:`BackendResponse`, and both use the *same* prompt and the
*same* JSON schema: the Agent SDK's ``output_format`` mirrors the Messages API's
``output_config.format``, so the schema is hard-enforced either way rather than being
a request the model may ignore.

**Cost note for the claude-code backend.** Each call is a fresh CLI invocation, but the
prompt cache is account-scoped rather than session-scoped, so the identical system
prefix is reused across calls. Measured on this pipeline: the first document costs
about $0.15 and creates ~13.7K cache tokens; every subsequent document reads that cache
and costs about $0.02-0.03. That 5-7x drop is the entire reason the taxonomy is
rendered deterministically -- one changed byte and every document pays full price
again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .judge import JudgeConfig, build_request_params, build_schema, build_system_blocks, build_user_content
from .taxonomy import Taxonomy


class BackendError(RuntimeError):
    pass


@dataclass
class BackendResponse:
    """One classification attempt, however it was obtained."""

    payload: dict[str, Any] | None = None
    error: str | None = None
    refusal: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    model: str = ""
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.payload is not None


class Backend(Protocol):
    name: str

    def classify(
        self, text: str, taxonomy: Taxonomy, config: JudgeConfig
    ) -> BackendResponse: ...


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


class AnthropicBackend:
    """Messages API backend. Also the only backend that supports batching."""

    name = "anthropic"

    def __init__(self, client: Any | None = None, api_key: str | None = None):
        self._client = client if client is not None else _make_anthropic_client(api_key)

    @property
    def client(self) -> Any:
        return self._client

    def classify(self, text: str, taxonomy: Taxonomy, config: JudgeConfig) -> BackendResponse:
        import json

        params, truncated = build_request_params(text, taxonomy, config)
        message = self._client.with_options(max_retries=config.max_retries).messages.create(
            **params
        )

        usage = _anthropic_usage(message)
        stop_reason = getattr(message, "stop_reason", None)

        # Check stop_reason before touching content: a refusal can carry an empty
        # content list, and indexing it blindly raises instead of reporting.
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            return BackendResponse(
                refusal=getattr(details, "category", None) or "refusal",
                usage=usage,
                model=config.model,
                truncated=truncated,
            )
        if stop_reason == "max_tokens":
            return BackendResponse(
                error="response hit max_tokens; JSON is incomplete. Raise max_tokens "
                "or lower max_document_chars.",
                usage=usage,
                model=config.model,
                truncated=truncated,
            )

        text_block = next(
            (b.text for b in message.content if getattr(b, "type", None) == "text"), None
        )
        if text_block is None:
            return BackendResponse(
                error="response contained no text block",
                usage=usage,
                model=config.model,
                truncated=truncated,
            )
        try:
            payload = json.loads(text_block)
        except json.JSONDecodeError as exc:
            return BackendResponse(
                error=f"could not parse judge output: {exc}",
                usage=usage,
                model=config.model,
                truncated=truncated,
            )

        return BackendResponse(
            payload=payload, usage=usage, model=config.model, truncated=truncated
        )


def _make_anthropic_client(api_key: str | None = None) -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BackendError(
            "the anthropic backend requires the anthropic SDK:\n"
            "  uv pip install -e 'datax[judge]'"
        ) from exc
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _anthropic_usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


# ---------------------------------------------------------------------------
# Claude Code agent
# ---------------------------------------------------------------------------


class ClaudeCodeBackend:
    """Backend that drives the local Claude Code agent.

    The agent is stripped down to a single-turn classifier:

    * ``system_prompt`` **replaces** Claude Code's default prompt rather than
      appending to it -- the default is ~38K tokens of coding-agent instructions and
      tool schemas, none of which a document classifier needs. Replacing it took a
      trivial call from $0.23 to a fraction of that.
    * ``tools=[]`` -- the judge reads one document and returns JSON; giving it a
      filesystem would be an unnecessary capability and more tokens.
    * ``setting_sources=[]`` -- project ``CLAUDE.md``, settings and skills are not
      loaded. Besides the token cost, loading them would make the cached prefix depend
      on the checkout it runs in, so results would drift between machines.
    * ``max_turns=1`` -- one question, one answer; no agentic loop.
    """

    name = "claude-code"

    def __init__(self, *, cwd: str | None = None):
        self._cwd = cwd
        self._checked = False

    def _require_sdk(self):
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BackendError(
                "the claude-code backend requires claude-agent-sdk and a working "
                "`claude` CLI:\n"
                "  uv pip install -e 'datax[agent]'\n"
                "  (and install Claude Code: https://claude.com/claude-code)"
            ) from exc
        return claude_agent_sdk

    def build_options(self, taxonomy: Taxonomy, config: JudgeConfig):
        sdk = self._require_sdk()
        # cache=False: the cache_control marker is a Messages API concept. Claude Code
        # manages caching itself, and it works here precisely because this prompt
        # renders to identical bytes every time.
        system = "\n\n".join(
            block["text"] for block in build_system_blocks(taxonomy, cache=False)
        )
        options = dict(
            system_prompt=system,
            tools=[],
            setting_sources=[],
            max_turns=1,
            model=config.model,
            effort=config.effort,
            output_format={"type": "json_schema", "schema": build_schema(taxonomy)},
        )
        if config.max_cost_usd:
            options["max_budget_usd"] = config.max_cost_usd
        if self._cwd:
            options["cwd"] = self._cwd
        return sdk.ClaudeAgentOptions(**options)

    async def classify_async(
        self, text: str, taxonomy: Taxonomy, config: JudgeConfig
    ) -> BackendResponse:
        sdk = self._require_sdk()
        content, truncated = build_user_content(text, config)
        options = self.build_options(taxonomy, config)

        result = None
        async for message in sdk.query(prompt=content, options=options):
            if isinstance(message, sdk.ResultMessage):
                result = message

        if result is None:
            return BackendResponse(
                error="claude-code returned no result message",
                model=config.model,
                truncated=truncated,
            )

        usage = _claude_code_usage(result.usage or {})
        cost = result.total_cost_usd or 0.0

        if result.stop_reason == "refusal":
            return BackendResponse(
                refusal="refusal", usage=usage, cost_usd=cost, model=config.model,
                truncated=truncated,
            )
        if result.is_error:
            detail = "; ".join(result.errors or []) or result.subtype
            status = f" (HTTP {result.api_error_status})" if result.api_error_status else ""
            return BackendResponse(
                error=f"claude-code error: {detail}{status}",
                usage=usage, cost_usd=cost, model=config.model, truncated=truncated,
            )

        payload = result.structured_output
        if payload is None and result.result:
            # Fall back to parsing the text, in case structured output was not applied.
            import json

            try:
                payload = json.loads(result.result)
            except json.JSONDecodeError:
                payload = None
        if not isinstance(payload, dict):
            return BackendResponse(
                error=f"claude-code returned no structured output "
                f"(terminal_reason={result.terminal_reason!r})",
                usage=usage, cost_usd=cost, model=config.model, truncated=truncated,
            )

        return BackendResponse(
            payload=payload, usage=usage, cost_usd=cost, model=config.model,
            truncated=truncated,
        )

    def classify(self, text: str, taxonomy: Taxonomy, config: JudgeConfig) -> BackendResponse:
        """Synchronous wrapper.

        Raises if called from inside a running event loop; use
        :meth:`classify_async` there.
        """
        import anyio

        return anyio.run(self.classify_async, text, taxonomy, config)


def _claude_code_usage(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------


def make_backend(name: str, **kwargs) -> Backend:
    if name == "anthropic":
        return AnthropicBackend(**kwargs)
    if name in ("claude-code", "claude_code"):
        return ClaudeCodeBackend(**{k: v for k, v in kwargs.items() if k == "cwd"})
    raise BackendError(f"unknown backend {name!r}; expected 'anthropic' or 'claude-code'")
