"""Command line interface.

    python -m datax taxonomy                 inspect and validate the taxonomies
    python -m datax fetch nemotron           render gold-labelled .docx from Nemotron-PII
    python -m datax fetch corpus             download real .docx via the docx-corpus index
    python -m datax bulk                     bulk-download the corpus into topic/type folders
    python -m datax judge <dir>              classify .docx files into a manifest
    python -m datax validate <manifest>      check a manifest for internal consistency
    python -m datax stats <manifest>         dataset statistics and coverage gaps
    python -m datax evaluate <gold> <pred>   score judged labels against gold
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .extract import DocxReadError, extract_docx, looks_like_docx
from .manifest import SourceInfo, read_manifest, validate_manifest, write_manifest
from .taxonomy import default_taxonomy

DEFAULT_DOC_DIR = "datax/data/documents"
DEFAULT_MANIFEST = "datax/data/manifest.jsonl"
DEFAULT_GOLD = "datax/data/gold.jsonl"


# ---------------------------------------------------------------------------


def cmd_taxonomy(args: argparse.Namespace) -> int:
    taxonomy = default_taxonomy()
    if args.json:
        payload = {
            "industry_version": taxonomy.industry_version,
            "pii_version": taxonomy.pii_version,
            "industries": {
                ind.id: {
                    "label": ind.label,
                    "categories": {
                        cat.id: [s.id for s in cat.subcategories] for cat in ind.categories
                    },
                }
                for ind in taxonomy.industries
            },
            "pii_labels": taxonomy.pii_ids,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"industry taxonomy v{taxonomy.industry_version}")
    total_subs = 0
    for ind in taxonomy.industries:
        subs = taxonomy.subcategories(ind.id)
        total_subs += len(subs)
        print(f"  {ind.id:12s} {len(ind.categories):2d} categories, {len(subs):3d} subcategories")
    print(f"  {'TOTAL':12s} {total_subs:3d} subcategories across {len(taxonomy.industries)} industries")
    print()
    print(f"PII taxonomy v{taxonomy.pii_version}: {len(taxonomy.pii_labels)} labels")
    groups: dict[str, int] = {}
    for label in taxonomy.pii_labels:
        groups[label.group] = groups.get(label.group, 0) + 1
    for group, count in groups.items():
        print(f"  {group:16s} {count:2d}")
    phi = sum(1 for p in taxonomy.pii_labels if p.phi)
    special = sum(1 for p in taxonomy.pii_labels if p.special_category)
    print(f"  ({phi} PHI-relevant, {special} special-category)")

    crosswalk = taxonomy.nemotron_crosswalk()
    print()
    print(f"Nemotron document_type crosswalk entries: {len(crosswalk)}")
    return 0


# ---------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    taxonomy = default_taxonomy()

    if args.provider == "nemotron":
        from .sources import nemotron

        report, records = nemotron.fetch(
            out_dir,
            split=args.split,
            per_domain=args.per_group,
            cache_dir=args.cache_dir,
            taxonomy=taxonomy,
        )
        print(report.summary())
        if records:
            written = write_manifest(args.gold, records)
            print(f"wrote {written} gold record(s) to {args.gold}")
        return 0 if report.written else 1

    if args.provider == "corpus":
        from .sources import docxcorpus

        report, provenance = docxcorpus.fetch(
            out_dir,
            per_industry=args.per_group,
            cache_dir=args.cache_dir,
            delay_seconds=args.delay,
        )
        print(report.summary())
        # Persist provenance so `judge` can attribute each file to its index entry.
        prov_path = Path(args.cache_dir) / "corpus-provenance.json"
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        prov_path.write_text(
            json.dumps(
                {path: entry.__dict__ for path, entry in provenance.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote provenance for {len(provenance)} file(s) to {prov_path}")
        return 0 if report.written else 1

    print(f"unknown provider {args.provider!r}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------


def _load_provenance(cache_dir: str) -> dict[str, dict]:
    path = Path(cache_dir) / "corpus-provenance.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_gold_provenance(gold_path: str) -> dict[str, SourceInfo]:
    """Recover source attribution for documents that already have gold records.

    The gold manifest records where each rendered document came from, keyed by content
    hash. Reusing it here keeps judged records attributed to ``nemotron`` (and marked
    synthetic) instead of falling back to ``local``, which would misreport the corpus
    as containing real personal data.
    """
    path = Path(gold_path)
    if not path.exists():
        return {}
    try:
        return {r.file.sha256: r.source for r in read_manifest(path)}
    except Exception:  # noqa: BLE001 - provenance is best effort
        return {}


def _source_for(
    path: Path,
    sha256: str,
    corpus_provenance: dict[str, dict],
    gold_provenance: dict[str, SourceInfo],
) -> SourceInfo:
    known = gold_provenance.get(sha256)
    if known is not None:
        return known

    entry = corpus_provenance.get(str(path))
    if entry:
        from .sources.docxcorpus import DATASET, LICENSE

        return SourceInfo(
            provider="docxcorpus",
            reference=entry.get("url", f"{DATASET}#{entry.get('id', '')}"),
            license=LICENSE,
            synthetic=False,
        )
    return SourceInfo(provider="local", reference=str(path), license="unknown", synthetic=False)


def cmd_judge(args: argparse.Namespace) -> int:
    from .backends import BackendError, make_backend
    from .judge import BatchItem, JudgeConfig, collect_batch, judge_document, submit_batch

    taxonomy = default_taxonomy()
    config = JudgeConfig(
        model=args.model,
        effort=args.effort,
        max_document_chars=args.max_chars,
        occurrence_mode=args.occurrences,
        max_cost_usd=args.max_cost_per_doc,
    )

    if args.batch and args.backend != "anthropic":
        print(
            "--batch requires --backend anthropic; the Batches API is a Messages API "
            "feature and Claude Code has no equivalent.",
            file=sys.stderr,
        )
        return 2

    root = Path(args.input)
    paths = sorted(p for p in root.rglob("*.docx") if p.is_file() and not p.name.startswith("~$"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no .docx files found under {root}", file=sys.stderr)
        return 1

    # Skip anything already in the manifest, so a re-run resumes instead of re-paying.
    already: set[str] = set()
    manifest_path = Path(args.manifest)
    if manifest_path.exists() and not args.overwrite:
        already = {r.file.sha256 for r in read_manifest(manifest_path)}

    provenance = _load_provenance(args.cache_dir)
    gold_provenance = _load_gold_provenance(args.gold)
    pending = []
    for path in paths:
        if not looks_like_docx(path):
            print(f"  skip (not a docx): {path}", file=sys.stderr)
            continue
        try:
            extracted = extract_docx(path)
        except DocxReadError as exc:
            print(f"  skip ({exc}): {path}", file=sys.stderr)
            continue
        if extracted.sha256 in already:
            continue
        if extracted.is_empty:
            print(f"  skip (no extractable text): {path}", file=sys.stderr)
            continue
        pending.append((path, extracted))

    print(
        f"{len(pending)} document(s) to judge ({len(paths) - len(pending)} skipped) "
        f"via backend {args.backend!r}"
    )
    if not pending:
        return 0
    if args.dry_run:
        for path, extracted in pending[:10]:
            print(f"  would judge {path} ({extracted.word_count} words)")
        return 0

    # Truncating up front keeps the incremental append below simple and correct.
    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    try:
        backend = make_backend(args.backend)
    except BackendError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    records = []
    failures = 0
    total_cost = 0.0

    if args.batch:
        client = backend.client
        items = [
            BatchItem(
                custom_id=f"doc-{index}",
                extracted=extracted,
                source=_source_for(path, extracted.sha256, provenance, gold_provenance),
            )
            for index, (path, extracted) in enumerate(pending)
        ]
        batch_id = submit_batch(client, items, taxonomy=taxonomy, config=config)
        print(f"submitted batch {batch_id}; polling (most batches finish within an hour)")
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            counts = batch.request_counts
            print(f"  status={batch.processing_status} processing={counts.processing}")
            time.sleep(args.poll_seconds)
        outcomes = collect_batch(client, batch_id, items, taxonomy=taxonomy, config=config)
        for item in items:
            outcome = outcomes.get(item.custom_id)
            if outcome is None or not outcome.ok:
                failures += 1
                reason = outcome.error or outcome.refusal if outcome else "no result"
                print(f"  ! {item.extracted.path}: {reason}", file=sys.stderr)
                continue
            records.append(outcome.record)
    else:
        for index, (path, extracted) in enumerate(pending, start=1):
            source = _source_for(path, extracted.sha256, provenance, gold_provenance)
            outcome = judge_document(
                backend, extracted, source, taxonomy=taxonomy, config=config
            )
            total_cost += outcome.cost_usd
            if not outcome.ok:
                failures += 1
                print(f"  ! {path}: {outcome.error or outcome.refusal}", file=sys.stderr)
                continue
            records.append(outcome.record)
            record = outcome.record
            cached = outcome.usage.get("cache_read_input_tokens", 0)
            print(
                f"  [{index}/{len(pending)}] {path.name}: "
                f"{record.industry.id}/{record.industry.subcategory} "
                f"({len(record.pii.labels)} PII label(s), "
                f"{len(record.spans)} span(s))"
                + (f"  ${outcome.cost_usd:.4f}" if outcome.cost_usd else "")
                + (f" cache_read={cached}" if cached else "")
            )
            # Append as we go: a run that dies at document 80 must not throw away the
            # 79 classifications already paid for.
            write_manifest(args.manifest, [outcome.record], append=True)

    written = len(records)
    if args.batch:
        written = write_manifest(args.manifest, records, append=True)
    print(f"wrote {written} record(s) to {args.manifest} ({failures} failure(s))")
    if total_cost:
        print(f"total cost: ${total_cost:.2f} (${total_cost / max(written, 1):.4f}/document)")
    return 0 if written else 1


# ---------------------------------------------------------------------------


def cmd_bulk(args: argparse.Namespace) -> int:
    from .sources import bulk, docxcorpus

    out_dir = Path(args.out)
    index_path = docxcorpus.download_index(args.cache_dir)

    topics = bulk.ALL_TOPICS if args.topics == "all" else [t.strip() for t in args.topics.split(",")]
    types = [t.strip() for t in args.types.split(",")] if args.types else None

    if args.retry_failed:
        urls = set(bulk.load_failures(out_dir))
        if not urls:
            print(f"no failures recorded in {out_dir / bulk.FAILURES_FILE}", file=sys.stderr)
            return 1
        entries = [
            e
            for e in bulk.select_all(
                index_path, topics=topics, language=None, min_confidence=0.0
            )
            if e.url in urls
        ]
        print(f"retrying {len(entries):,} previously failed download(s)")
    else:
        entries = bulk.select_all(
            index_path,
            topics=topics,
            language=args.lang or None,
            types=types,
            min_confidence=args.min_confidence,
            min_words=args.min_words,
            limit=args.limit or None,
        )

    if not entries:
        print("no index rows matched those filters", file=sys.stderr)
        return 1

    grid = bulk.summarise_plan(entries)
    print(f"{len(entries):,} document(s) selected -> {out_dir}/<topic>/<type>/<id>.docx\n")
    print(f"{'topic':<14}{'type':<18}{'count':>10}")
    print("-" * 42)
    for topic in sorted(grid):
        for doc_type, count in sorted(grid[topic].items(), key=lambda kv: -kv[1]):
            print(f"{topic:<14}{doc_type:<18}{count:>10,}")
        print(f"{topic:<14}{'(all types)':<18}{sum(grid[topic].values()):>10,}")
        print()

    estimated = bulk.estimate_bytes(entries) if not args.no_estimate else 0
    free = bulk.free_bytes(out_dir)
    if estimated:
        print(f"estimated download size: {estimated / 1e9:.1f} GB (sampled from live HEAD requests)")
    print(f"free disk at destination: {free / 1e9:.1f} GB")
    if estimated and estimated > free * 0.9:
        print(
            f"\nrefusing to start: {estimated / 1e9:.1f} GB needed but only "
            f"{free / 1e9:.1f} GB free. Narrow the selection or point --out at a "
            "bigger volume.",
            file=sys.stderr,
        )
        return 1

    if args.plan:
        print("\n--plan given; nothing downloaded.")
        return 0

    hours = len(entries) / max(args.workers, 1) * 0.35 / 3600
    print(f"\nstarting with {args.workers} workers (rough estimate: {hours:.1f} h)")
    print("safe to interrupt and re-run -- existing files are skipped.\n")

    def progress(report: bulk.BulkReport) -> None:
        done = report.downloaded + report.skipped_existing + report.failed
        pct = done / report.requested * 100 if report.requested else 0
        rate = report.downloaded / report.elapsed_seconds if report.elapsed_seconds else 0
        remaining = (report.requested - done) / rate / 60 if rate else 0
        print(
            f"  {done:>7,}/{report.requested:,} ({pct:5.1f}%)  "
            f"{report.bytes_written / 1e9:5.2f} GB  {rate:5.1f} files/s  "
            f"~{remaining:.0f} min left  ({report.failed} failed)"
        )

    report = bulk.download_all(
        entries,
        out_dir,
        workers=args.workers,
        retries=args.retries,
        timeout=args.timeout,
        verify_existing=args.verify_existing,
        on_progress=progress,
        progress_every=args.progress_every,
    )
    print()
    print(report.summary())
    print(f"\nindex:    {out_dir / bulk.INDEX_FILE}")
    if report.failed:
        print(f"failures: {out_dir / bulk.FAILURES_FILE}  (re-run with --retry-failed)")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    taxonomy = default_taxonomy()
    report = validate_manifest(args.manifest, taxonomy)
    print(f"{report.valid}/{report.total} record(s) valid")
    if report.duplicate_uids:
        print(f"duplicate uids: {len(report.duplicate_uids)}")
    if report.duplicate_sha256:
        print(f"duplicate file hashes: {len(report.duplicate_sha256)}")
    for uid, problems in list(report.problems.items())[: args.max_report]:
        print(f"\n{uid}:")
        for problem in problems:
            print(f"  - {problem}")
    if len(report.problems) > args.max_report:
        print(f"\n... and {len(report.problems) - args.max_report} more record(s) with problems")
    return 0 if report.ok else 1


def cmd_stats(args: argparse.Namespace) -> int:
    from . import stats as stats_module

    taxonomy = default_taxonomy()
    stats = stats_module.compute(read_manifest(args.manifest), taxonomy)
    if args.json:
        print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(stats.summary())
    if args.out:
        stats_module.write_json(args.out, stats)
        print(f"\nwrote {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_files

    report = evaluate_files(args.gold, args.predicted)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
    if report.matched == 0:
        print(
            "\nno documents matched between gold and predicted manifests "
            "(they are joined on file.sha256)",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datax", description="Build a classified .docx dataset for ML file classification."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tax = sub.add_parser("taxonomy", help="inspect and validate the taxonomies")
    p_tax.add_argument("--json", action="store_true")
    p_tax.set_defaults(func=cmd_taxonomy)

    p_fetch = sub.add_parser("fetch", help="download or render source documents")
    p_fetch.add_argument("provider", choices=["nemotron", "corpus"])
    p_fetch.add_argument("--out", default=DEFAULT_DOC_DIR)
    p_fetch.add_argument(
        "--per-group",
        type=int,
        default=25,
        help="documents per industry (corpus) or per Nemotron domain",
    )
    p_fetch.add_argument("--split", default="train", choices=["train", "test"])
    p_fetch.add_argument("--cache-dir", default="datax/data/cache")
    p_fetch.add_argument("--gold", default=DEFAULT_GOLD, help="where to write gold records")
    p_fetch.add_argument("--delay", type=float, default=0.25, help="seconds between downloads")
    p_fetch.set_defaults(func=cmd_fetch)

    p_judge = sub.add_parser("judge", help="classify .docx files with the LLM judge")
    p_judge.add_argument("input", nargs="?", default=DEFAULT_DOC_DIR)
    p_judge.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p_judge.add_argument(
        "--backend",
        default="claude-code",
        choices=["claude-code", "anthropic"],
        help="claude-code drives the local Claude Code agent (no API key needed); "
        "anthropic calls the Messages API and supports --batch",
    )
    p_judge.add_argument(
        "--max-cost-per-doc",
        type=float,
        default=0.0,
        help="per-document spend ceiling for the claude-code backend (0 = no limit)",
    )
    p_judge.add_argument("--model", default="claude-opus-5")
    p_judge.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p_judge.add_argument("--max-chars", type=int, default=60_000)
    p_judge.add_argument("--occurrences", default="all", choices=["all", "first"])
    p_judge.add_argument("--limit", type=int, default=0)
    p_judge.add_argument("--batch", action="store_true", help="use the Batches API (50%% cheaper)")
    p_judge.add_argument("--poll-seconds", type=int, default=60)
    p_judge.add_argument("--overwrite", action="store_true", help="replace the manifest")
    p_judge.add_argument("--dry-run", action="store_true", help="list work without calling the API")
    p_judge.add_argument("--cache-dir", default="datax/data/cache")
    p_judge.add_argument(
        "--gold",
        default=DEFAULT_GOLD,
        help="gold manifest, read only to recover source attribution by content hash",
    )
    p_judge.set_defaults(func=cmd_judge)

    p_bulk = sub.add_parser(
        "bulk", help="bulk-download the docx-corpus into a <topic>/<type>/<id>.docx tree"
    )
    p_bulk.add_argument("--out", default="datax/data/corpus", help="destination root")
    p_bulk.add_argument(
        "--topics",
        default="government,finance,healthcare",
        help="comma-separated corpus topics, or 'all' for every topic",
    )
    p_bulk.add_argument("--lang", default="en", help="ISO 639-1 language, empty for all")
    p_bulk.add_argument("--types", default="", help="comma-separated document types (default: all)")
    p_bulk.add_argument("--min-confidence", type=float, default=0.0)
    p_bulk.add_argument("--min-words", type=int, default=0)
    p_bulk.add_argument("--limit", type=int, default=0)
    p_bulk.add_argument(
        "--workers", type=int, default=8, help="parallel downloads (the corpus suggests 4)"
    )
    p_bulk.add_argument("--retries", type=int, default=3)
    p_bulk.add_argument("--timeout", type=int, default=30)
    p_bulk.add_argument("--progress-every", type=int, default=250)
    p_bulk.add_argument(
        "--verify-existing",
        action="store_true",
        help="re-open already-downloaded files to confirm they are valid .docx",
    )
    p_bulk.add_argument("--retry-failed", action="store_true", help="re-attempt failures.jsonl")
    p_bulk.add_argument("--plan", action="store_true", help="show the plan and exit")
    p_bulk.add_argument("--no-estimate", action="store_true", help="skip the HEAD size sample")
    p_bulk.add_argument("--cache-dir", default="datax/data/cache")
    p_bulk.set_defaults(func=cmd_bulk)

    p_val = sub.add_parser("validate", help="check a manifest for internal consistency")
    p_val.add_argument("manifest", nargs="?", default=DEFAULT_MANIFEST)
    p_val.add_argument("--max-report", type=int, default=20)
    p_val.set_defaults(func=cmd_validate)

    p_stats = sub.add_parser("stats", help="dataset statistics and coverage gaps")
    p_stats.add_argument("manifest", nargs="?", default=DEFAULT_MANIFEST)
    p_stats.add_argument("--json", action="store_true")
    p_stats.add_argument("--out", help="also write statistics to this JSON file")
    p_stats.set_defaults(func=cmd_stats)

    p_eval = sub.add_parser("evaluate", help="score judged labels against gold labels")
    p_eval.add_argument("gold", nargs="?", default=DEFAULT_GOLD)
    p_eval.add_argument("predicted", nargs="?", default=DEFAULT_MANIFEST)
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
