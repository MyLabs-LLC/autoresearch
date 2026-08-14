"""Loading and validation for the industry and PII taxonomies.

Both taxonomies are data, not code: they live in ``datax/taxonomy/*.json`` so they can
be versioned, diffed, and shipped alongside the dataset. This module is the only place
that knows their on-disk shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

def _find_taxonomy_dir() -> Path:
    """Locate the taxonomy directory in both the source tree and an installed wheel.

    Source layout puts it at ``datax/taxonomy`` (two levels above this module);
    the wheel ships it inside the package as ``datax/taxonomy``.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent / "taxonomy", here.parents[2] / "taxonomy"):
        if (candidate / "industries.json").exists():
            return candidate
    # Fall back to the source-layout path so the error names a useful location.
    return here.parents[2] / "taxonomy"


TAXONOMY_DIR = _find_taxonomy_dir()

# Industry assigned to a document that belongs to none of the three target industries.
# The judge needs this escape hatch: without it, every out-of-scope document is forced
# into a wrong label and the precision numbers become meaningless.
OTHER_INDUSTRY = "other"


class TaxonomyError(ValueError):
    """Raised when a taxonomy file is structurally invalid."""


@dataclass(frozen=True)
class PiiLabel:
    id: str
    group: str
    sensitivity: str
    description: str
    phi: bool = False
    special_category: bool = False


@dataclass(frozen=True)
class Subcategory:
    id: str
    label: str
    description: str
    industry: str
    category: str
    nemotron_document_types: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        """Fully qualified id, e.g. ``healthcare/clinical_records/pathology_report``."""
        return f"{self.industry}/{self.category}/{self.id}"


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    industry: str
    subcategories: tuple[Subcategory, ...]


@dataclass(frozen=True)
class Industry:
    id: str
    label: str
    description: str
    nemotron_domains: tuple[str, ...]
    categories: tuple[Category, ...]


@dataclass
class Taxonomy:
    industry_version: str
    pii_version: str
    industries: tuple[Industry, ...]
    pii_labels: tuple[PiiLabel, ...]

    _by_industry: dict[str, Industry] = field(default_factory=dict, repr=False)
    _pii_by_id: dict[str, PiiLabel] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_industry = {i.id: i for i in self.industries}
        self._pii_by_id = {p.id: p for p in self.pii_labels}

    # -- industries ---------------------------------------------------------

    @property
    def industry_ids(self) -> list[str]:
        """Industry ids the judge may choose from, including the ``other`` escape hatch."""
        return [i.id for i in self.industries] + [OTHER_INDUSTRY]

    def industry(self, industry_id: str) -> Industry:
        try:
            return self._by_industry[industry_id]
        except KeyError:
            raise TaxonomyError(f"unknown industry {industry_id!r}") from None

    def subcategories(self, industry_id: str | None = None) -> list[Subcategory]:
        out: list[Subcategory] = []
        for ind in self.industries:
            if industry_id is not None and ind.id != industry_id:
                continue
            for cat in ind.categories:
                out.extend(cat.subcategories)
        return out

    def subcategory_ids(self, industry_id: str | None = None) -> list[str]:
        """Leaf ids. Not globally unique -- ``tax_return`` exists under finance and
        government -- so they are only meaningful together with an industry."""
        return [s.id for s in self.subcategories(industry_id)]

    def find_subcategory(self, industry_id: str, subcategory_id: str) -> Subcategory | None:
        for sub in self.subcategories(industry_id):
            if sub.id == subcategory_id:
                return sub
        return None

    def nemotron_crosswalk(self) -> dict[str, Subcategory]:
        """Map a Nemotron-PII ``document_type`` string to our leaf subcategory.

        Keys are casefolded because Nemotron's own values are inconsistently cased
        (``Tax Return`` vs ``blood test report``).
        """
        table: dict[str, Subcategory] = {}
        for sub in self.subcategories():
            for doc_type in sub.nemotron_document_types:
                table.setdefault(doc_type.casefold(), sub)
        return table

    def industry_for_nemotron_domain(self, domain: str) -> str | None:
        """Reverse of ``Industry.nemotron_domains``.

        Ambiguous domains (``Insurance`` maps to both healthcare and finance) resolve to
        the first industry that declares them, so callers that need an exact mapping
        should use the document-type crosswalk instead.
        """
        for ind in self.industries:
            if domain in ind.nemotron_domains:
                return ind.id
        return None

    # -- PII ----------------------------------------------------------------

    @property
    def pii_ids(self) -> list[str]:
        return [p.id for p in self.pii_labels]

    def pii(self, label_id: str) -> PiiLabel:
        try:
            return self._pii_by_id[label_id]
        except KeyError:
            raise TaxonomyError(f"unknown PII label {label_id!r}") from None

    def max_sensitivity(self, label_ids: list[str]) -> str:
        """Highest sensitivity across the given labels; ``none`` for an empty list."""
        order = ["none", "low", "medium", "high", "critical"]
        best = "none"
        for label_id in label_ids:
            level = self._pii_by_id[label_id].sensitivity
            if order.index(level) > order.index(best):
                best = level
        return best


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise TaxonomyError(f"taxonomy file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_taxonomy(taxonomy_dir: Path | None = None) -> Taxonomy:
    """Load and validate both taxonomy files. Raises :class:`TaxonomyError` on any
    structural problem -- duplicate ids, missing fields, empty categories."""
    root = taxonomy_dir or TAXONOMY_DIR
    industries_raw = _load_json(root / "industries.json")
    pii_raw = _load_json(root / "pii.json")

    pii_labels: list[PiiLabel] = []
    seen_pii: set[str] = set()
    groups = pii_raw.get("groups", {})
    for entry in pii_raw["labels"]:
        label_id = entry["id"]
        if label_id in seen_pii:
            raise TaxonomyError(f"duplicate PII label id {label_id!r}")
        if entry["group"] not in groups:
            raise TaxonomyError(f"PII label {label_id!r} has undeclared group {entry['group']!r}")
        if entry["sensitivity"] not in pii_raw["sensitivity_levels"]:
            raise TaxonomyError(
                f"PII label {label_id!r} has undeclared sensitivity {entry['sensitivity']!r}"
            )
        seen_pii.add(label_id)
        pii_labels.append(
            PiiLabel(
                id=label_id,
                group=entry["group"],
                sensitivity=entry["sensitivity"],
                description=entry["description"],
                phi=bool(entry.get("phi", False)),
                special_category=bool(entry.get("special_category", False)),
            )
        )

    industries: list[Industry] = []
    for ind_raw in industries_raw["industries"]:
        ind_id = ind_raw["id"]
        if ind_id == OTHER_INDUSTRY:
            raise TaxonomyError(f"{OTHER_INDUSTRY!r} is reserved and cannot be an industry id")
        categories: list[Category] = []
        seen_sub: set[str] = set()
        for cat_raw in ind_raw["categories"]:
            subs: list[Subcategory] = []
            for sub_raw in cat_raw["subcategories"]:
                sub_id = sub_raw["id"]
                if sub_id in seen_sub:
                    raise TaxonomyError(
                        f"duplicate subcategory id {sub_id!r} within industry {ind_id!r}"
                    )
                seen_sub.add(sub_id)
                subs.append(
                    Subcategory(
                        id=sub_id,
                        label=sub_raw["label"],
                        description=sub_raw["description"],
                        industry=ind_id,
                        category=cat_raw["id"],
                        nemotron_document_types=tuple(sub_raw.get("nemotron_document_types", ())),
                    )
                )
            if not subs:
                raise TaxonomyError(f"category {ind_id}/{cat_raw['id']} has no subcategories")
            categories.append(
                Category(id=cat_raw["id"], label=cat_raw["label"], industry=ind_id, subcategories=tuple(subs))
            )
        industries.append(
            Industry(
                id=ind_id,
                label=ind_raw["label"],
                description=ind_raw["description"],
                nemotron_domains=tuple(ind_raw.get("nemotron_domains", ())),
                categories=tuple(categories),
            )
        )

    return Taxonomy(
        industry_version=industries_raw["version"],
        pii_version=pii_raw["version"],
        industries=tuple(industries),
        pii_labels=tuple(pii_labels),
    )


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    """Process-wide cached taxonomy. Cached so the judge's system prompt renders to
    identical bytes on every call, which is what keeps the prompt cache warm."""
    return load_taxonomy()
