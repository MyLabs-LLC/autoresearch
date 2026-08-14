# datax — a classified `.docx` dataset for ML file classification

A pipeline that builds a corpus of Word documents from **healthcare, finance and
government**, labelled with an industry taxonomy and a PII/PHI taxonomy, using
Claude as the judge — and then measures how good those labels actually are.

The PII vocabulary is byte-compatible with
[`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII), and the
manifest keeps that dataset's nine field names and value conventions, so a datax
manifest and Nemotron rows can be concatenated and consumed by the same code.

---

## What the numbers actually are

The brief asked for "151 industry tags and 60 PII tags, like the NVIDIA dataset".
Measured against the real dataset (both splits, 200,000 rows), the numbers are
different, and this project uses the measured ones:

| | Expected | Measured in Nemotron-PII | What datax ships |
|---|---|---|---|
| PII / PHI labels | 60 | **55** (identical in train and test) | **55**, the same names |
| Industry domains | 3 | **30** (`Healthcare`, `Finance`, `Government` among them) | **3** target industries |
| Document types | 151 | **free text**, 1,451 distinct across 30 domains — not a closed vocabulary | **162** closed leaf tags |

Details worth knowing:

- **55, not 60.** The dataset card says "55+ PII/PHI categories". Extracting every
  distinct `label` from every `spans` entry in both splits gives exactly 55, and the
  train and test vocabularies match each other exactly. `taxonomy/pii.json` uses those
  55 names verbatim — inventing five more would have broken drop-in compatibility,
  which is the whole point of matching NVIDIA.
- **Nemotron has no closed document-type vocabulary.** Its `document_type` is a
  free-text string. The three target domains contain 53 (healthcare), 53 (finance) and
  71 (government) distinct values — 176 in total, with near-duplicates like
  `Imaging Report` / `Medical Imaging Report` and `Pathology Report` /
  `Pathology Summary`. Normalising and de-duplicating those yields the **162** leaf
  tags in `taxonomy/industries.json`, each carrying a `nemotron_document_types`
  crosswalk back to the strings it covers. So the "151 tags" intuition was close; the
  exact figure just falls out of the de-duplication.

Run `python -m datax taxonomy` to print the live counts.

### Two things about Nemotron-PII worth knowing before you build on it

Both were found while building this, both are handled, and both will bite anyone who
assumes otherwise:

- **`uid` is not unique.** The train split has 100,000 rows over **50,000 distinct
  uids** — each document appears once per locale and the two variants share a uid,
  though their text differs (exactly one of the 50,000 pairs is identical). Keying on
  `uid` alone gives you duplicate manifest ids and, worse, silently overwrites one
  variant with the other on disk. datax keys on `uid-locale`, which is unique across
  all 100,000 rows.
- **1.53% of spans have a `text` field that doesn't match their own offsets.**
  12,594 of 825,456 train-split spans have `span["text"] != text[start:end]`. Two
  causes, both benign: normalised values (`'spanish'` where the document says
  `'Spanish'`), and numeric labels (`age`, `cvv`, `pin`) stored as `int` so `44 != "44"`.
  The **offsets are correct** in both cases, so datax repairs the text field to the
  actual substring and records a `gold_span_text_normalised` note, rather than
  discarding the span. Discarding would have thrown away ~2% of the gold labels.

---

## Taxonomies

**Industry** — three levels, `industry / category / subcategory`:

```
healthcare  9 categories   44 subcategories
finance     8 categories   53 subcategories
government 10 categories   65 subcategories
```

Leaf ids are unique *within* an industry, not globally — `tax_return` exists under
both finance and government. That is why the judge is constrained on the full path
(`finance/banking_and_lending/tax_return`) rather than on a bare id.

There is also an `other` escape hatch. Without it, every out-of-scope document gets
forced into a wrong label and the precision numbers stop meaning anything.

**PII/PHI** — the 55 Nemotron labels, grouped (identity, contact, location,
government_id, financial, health, biometric, credential, network, demographic,
temporal, organization) and annotated with a sensitivity level, a HIPAA `phi` flag,
and a GDPR Article 9 `special_category` flag. Those annotations are datax's addition;
the label *names* are NVIDIA's.

---

## Where the documents come from

Two sources, deliberately different in kind:

**`nemotron`** — renders Nemotron-PII rows for the Healthcare, Finance and Government
domains into real `.docx` files. Those rows are synthetic, CC-BY-4.0, and already carry
span-level PII annotations, which buys two things: a labelled corpus containing no real
personal data, and a **gold set** to score the judge against.

**`corpus`** — real documents from the public web, discovered through the
[`superdoc-dev/docx-corpus`](https://huggingface.co/datasets/superdoc-dev/docx-corpus)
index: 736K public `.docx` URLs pre-classified by type and topic, whose topic
vocabulary happens to include exactly `healthcare`, `finance` and `government`.

> A hardcoded list of agency URLs was tried first and abandoned. Guessed `.gov` paths
> 404 immediately, and several agencies (`hhs.gov`, `sec.gov`, `ftc.gov`, `epa.gov`)
> return 403 to non-browser clients. An index that is itself versioned and downloadable
> is reproducible; a handwritten URL list is link-rot waiting to happen. The corpus'
> own topic is kept only as provenance — documents still go through the judge, so the
> two labels can be compared rather than one being trusted.

`datax.sources.web` will still download an arbitrary caller-supplied URL list if you
have one.

---

## How labelling works

Three design decisions carry most of the weight:

**The judge never reports character offsets.** It reports a label plus the *verbatim
text* it saw; `datax.spans` then locates that quote in the document. Offsets are
correct by construction. A hallucinated quote simply fails to resolve and is recorded
as `unresolved_evidence` — there is no path that yields an offset pointing at the wrong
characters. (Models are unreliable at counting characters and reliable at quoting.)

**The subcategory is one enum, not three fields.** Structured outputs don't support
conditional schemas, so an industry-dependent vocabulary can't be expressed as
`industry` + `subcategory`. Encoding the leaf as a single `industry/category/subcategory`
path makes all three levels machine-enforced by one `enum`.

**The taxonomy sits before the cache breakpoint.** It's the large stable part of the
prompt; the document is the volatile part and goes after it. Rendering is deterministic
(sorted, no timestamps) because one changed byte in the prefix throws the cache away.
Watch `usage.cache_read_input_tokens` — if it stays at zero across documents, something
is invalidating the prefix.

Model is `claude-opus-5` with structured outputs and `effort` defaulting to `medium`.

### Two judge backends

| | `claude-code` (default) | `anthropic` |
|---|---|---|
| Credentials | none — reuses Claude Code's own auth | `ANTHROPIC_API_KEY` or `ant auth login` |
| Transport | `claude-agent-sdk` → local `claude` CLI | Messages API |
| Schema enforcement | `output_format` json_schema | `output_config.format` json_schema |
| Batching | no | yes, `--batch`, 50% cheaper |
| Measured cost | ~$0.15 first doc, ~$0.02–0.05 after | lower, especially batched |

Both send the *same* prompt and the *same* schema. The Agent SDK's `output_format`
mirrors the Messages API's `output_config.format`, so the enum is hard-enforced either
way rather than being a request the model may ignore.

The `claude-code` backend strips the agent down to a single-turn classifier, which
matters more than it sounds:

- **`system_prompt` replaces** Claude Code's default instead of appending to it. The
  default is ~38K tokens of coding-agent instructions and tool schemas — a trivial
  `claude -p "Say OK"` in this container cost **$0.23** because of it.
- **`tools=[]`** — the judge reads one document and returns JSON. A filesystem would be
  an unnecessary capability and more tokens.
- **`setting_sources=[]`** — project `CLAUDE.md`, settings and skills are not loaded.
  Beyond token cost, loading them would make the cached prefix depend on which checkout
  it runs in, so results would drift between machines.
- **`max_turns=1`** — one question, one answer, no agentic loop.

**The prompt cache works across separate CLI invocations**, because it is account-scoped
rather than session-scoped. Measured here: first document $0.151 with 13.7K cache-creation
tokens, every document after it `cache_read=13305` at $0.02–0.05. That 5–7× drop is
the entire payoff of rendering the taxonomy deterministically — one changed byte and
every document pays full price again.

Use `anthropic --batch` when building a large corpus and `claude-code` when you want it
to just work without provisioning a key.

---

## Install

```bash
cd datax
uv venv && uv pip install -e '.[all,dev]'
```

No API key is needed with the default `claude-code` backend — it uses Claude Code's own
credentials. For the `anthropic` backend, set `ANTHROPIC_API_KEY` or run `ant auth login`.

Reading `.docx` and building manifests needs no third-party parser — only `pyarrow`,
for the source indexes. Extras: `docx` to *write* documents (the `nemotron` source),
`agent` for the Claude Code backend, `judge` for the Messages API backend.

## Use

```bash
python -m datax taxonomy                       # counts, groups, crosswalk size
python -m datax fetch nemotron --per-group 40  # gold-labelled .docx  -> data/gold.jsonl
python -m datax fetch corpus  --per-group 40   # real .docx from the public web
python -m datax judge                          # classify everything -> data/manifest.jsonl
python -m datax validate                       # internal consistency
python -m datax stats --out data/stats.json    # distribution + coverage gaps
python -m datax evaluate                       # judge vs gold: accuracy, P/R/F1
```

`judge` defaults to the Claude Code backend. For the API instead:

```bash
python -m datax judge --backend anthropic --batch      # half price, no key-free path
python -m datax judge --max-cost-per-doc 0.10          # spend ceiling per document
```

`judge` is resumable: documents already present in the manifest (matched on content
hash) are skipped, so a re-run costs nothing for work already done. Use `--dry-run` to
see the workload without spending anything.

---

## Manifest format

One JSON object per line. The nine Nemotron-compatible fields keep their exact names:

```jsonc
{
  "uid": "...", "domain": "Healthcare", "document_type": "Discharge Summary",
  "document_description": "...", "document_format": "unstructured", "locale": "us",
  "text": "Patient: Jane Doe\nMRN: 88213-A",
  "spans": [{"start": 9, "end": 13, "text": "Jane", "label": "first_name"}],
  "text_tagged": "Patient: [Jane]first_name\nMRN: [88213-A]medical_record_number",

  // datax extensions, namespaced so the two vocabularies cannot collide
  "file":     {"path": "...", "sha256": "...", "size_bytes": 4211, "word_count": 6, ...},
  "source":   {"provider": "nemotron", "reference": "...", "license": "cc-by-4.0",
               "synthetic": true},
  "industry": {"id": "healthcare", "category": "admission_and_discharge",
               "subcategory": "discharge_summary", "confidence": 0.93, "rationale": "..."},
  "pii":      {"has_pii": true, "labels": ["first_name", "medical_record_number"],
               "count_by_label": {...}, "max_sensitivity": "critical",
               "contains_phi": true, "contains_special_category": true,
               "unresolved_evidence": []},
  "judge":    {"model": "claude-opus-5", "label_source": "llm_judge", ...}
}
```

`spans`, `text_tagged`, `pii` and `domain` are **derived** from the resolved spans and
the taxonomy, never taken on trust, so a record is internally consistent by
construction. `validate` re-checks all of it — in particular that every span's recorded
`text` still equals `text[start:end]`, which is the failure mode that silently ruins a
labelled dataset.

`label_source` separates `gold` records (Nemotron's own annotations) from `llm_judge`
records. Keep them in separate files if you plan to train on one and evaluate on the
other.

---

## Evaluation

`python -m datax evaluate` joins gold and predicted records on file content hash and
reports:

- industry accuracy, with a confusion matrix
- subcategory accuracy (only over documents where the crosswalk gives a gold leaf —
  otherwise you'd be measuring crosswalk coverage, not judge quality)
- PII **document level**: did the judge notice this document contains an `ssn` at all —
  the metric that matters for routing, DLP triage and file classification
- PII **span level**: exact offsets and label — the stricter metric, and the right one
  if you're training an NER model or a redactor
- micro *and* macro F1, because a micro average is dominated by `date` and
  `company_name` and hides how a rare-but-critical label like `ssn` is doing
- the five weakest labels by F1, which is where prompt work should go next

`stats` additionally reports **coverage gaps**: which PII labels and which
subcategories the dataset never exercises. That's the number that tells you what the
next batch of documents needs to contain.

---

## Notes and limits

- **No real personal data is generated by this project.** The `nemotron` documents are
  synthetic by construction. The `corpus` documents are real files from the public web
  and *may* contain real personal information — they are fetched by URL and not
  redistributed here. Review the licence and your own obligations before publishing a
  corpus built from them.
- **Truncation is explicit.** Documents longer than `--max-chars` (default 60,000) are
  truncated *for the judge*; spans still resolve against the full text, but PII past the
  cut is not seen. That is recorded as `unresolved_evidence` rather than hidden.
- **Headers and footers are detected but excluded** from `text`. They repeat per page
  and would corrupt the single linear offset space spans live in.
- **Untrusted input is treated as such.** Downloads are accepted only after the bytes
  prove to be a Word document (zip magic, then `word/document.xml`), sizes are capped
  while streaming, and XML parts carrying a `DOCTYPE` are refused outright.
- **`other` is a real answer.** A judge that never says `other` is overfitting to the
  three industries, and its precision numbers should not be believed.

## Status of this build

Verified end to end in a clean container, except the live API call (no credentials
were available there, so the judge was exercised through a stubbed client):

| Step | Result |
|---|---|
| `fetch nemotron --per-group 20` | 60/60 documents written, 0 failures |
| `.docx` round trip | **byte-exact for all 60** documents |
| gold spans preserved | **447/447 (100%)**, 4 repaired and noted |
| `validate` | 60/60 records valid |
| `fetch corpus --per-group 8` | 24/24 real documents downloaded, 0 failures |
| `evaluate` (gold vs gold) | 100% across every metric — evaluator sanity check |
| `pytest` | 102 passed |

The judge's outgoing request was checked offline: `claude-opus-5`, structured outputs
with a 163-value path enum and a 55-value label enum, a ~6.4K-token cached system
prefix (well above the 512-token minimum), and no `temperature`/`top_p`/`top_k` — all
of which that model rejects.

To run the judge for real you need credentials (`ANTHROPIC_API_KEY`, or `ant auth
login`). Start with `python -m datax judge --dry-run` to see the workload for free.

## Tests

```bash
python -m pytest tests -q
```

The suite covers taxonomy invariants, `.docx` round-trip fidelity (the property gold
offsets depend on), span resolution, manifest validation, and the judge's schema,
prompt-cache placement and response handling. The API call itself is the only part not
covered — everything around it is.
