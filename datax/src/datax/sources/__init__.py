"""Document sources.

Each source module turns some origin into .docx files on disk plus, where the origin
carries labels of its own, gold manifest records. They share one contract:

    fetch(out_dir, **options) -> FetchReport

Failures are collected, never raised: fetching a hundred documents from the open web
always produces some 404s and some HTML error pages served as .docx, and one bad URL
must not abort the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FetchReport:
    provider: str
    requested: int = 0
    written: int = 0
    skipped: int = 0
    failures: list[dict] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, reference: str, reason: str) -> None:
        self.failures.append({"reference": reference, "reason": reason})

    def summary(self) -> str:
        line = (
            f"{self.provider}: {self.written} written, {self.skipped} skipped, "
            f"{len(self.failures)} failed (of {self.requested} requested)"
        )
        return "\n".join([line, *(f"  ! {f['reference']}: {f['reason']}" for f in self.failures[:20])])
