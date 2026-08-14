"""The taxonomies are the dataset's contract; these tests pin the guarantees the
README makes about them."""

from __future__ import annotations

import json

import pytest

from datax.taxonomy import OTHER_INDUSTRY, TaxonomyError, default_taxonomy, load_taxonomy

# The measured Nemotron-PII span-label vocabulary. If this number changes, the
# compatibility claim in taxonomy/pii.json is no longer true.
NEMOTRON_LABEL_COUNT = 55


def test_taxonomy_loads():
    taxonomy = default_taxonomy()
    assert [i.id for i in taxonomy.industries] == ["healthcare", "finance", "government"]


def test_pii_vocabulary_matches_nemotron_exactly():
    taxonomy = default_taxonomy()
    assert len(taxonomy.pii_labels) == NEMOTRON_LABEL_COUNT
    assert len(set(taxonomy.pii_ids)) == NEMOTRON_LABEL_COUNT


def test_pii_labels_are_the_documented_set():
    # Spot-check labels that are easy to get subtly wrong (naming, not concept).
    ids = set(default_taxonomy().pii_ids)
    for label in [
        "ssn",
        "medical_record_number",
        "health_plan_beneficiary_number",
        "swift_bic",
        "http_cookie",
        "biometric_identifier",
        "certificate_license_number",
        "bank_routing_number",
    ]:
        assert label in ids, label
    # Plausible-but-wrong names that must NOT appear.
    for label in ["name", "address", "phone", "dob", "credit_card", "social_security_number"]:
        assert label not in ids, label


def test_subcategory_ids_unique_within_industry():
    taxonomy = default_taxonomy()
    for industry in taxonomy.industries:
        ids = taxonomy.subcategory_ids(industry.id)
        assert len(ids) == len(set(ids))


def test_leaf_paths_are_globally_unique():
    taxonomy = default_taxonomy()
    paths = [s.path for s in taxonomy.subcategories()]
    assert len(paths) == len(set(paths))


def test_tax_return_exists_in_two_industries():
    # Deliberate: leaf ids are only unique per industry, which is why the judge is
    # constrained on the full path rather than on the bare id.
    taxonomy = default_taxonomy()
    assert taxonomy.find_subcategory("finance", "tax_return") is not None
    assert taxonomy.find_subcategory("government", "tax_return") is not None


def test_other_is_not_a_real_industry():
    taxonomy = default_taxonomy()
    assert OTHER_INDUSTRY in taxonomy.industry_ids
    assert OTHER_INDUSTRY not in [i.id for i in taxonomy.industries]


def test_nemotron_crosswalk_is_case_insensitive():
    crosswalk = default_taxonomy().nemotron_crosswalk()
    # Nemotron mixes casing between document types; both forms must resolve.
    assert crosswalk["discharge summary".casefold()].id == "discharge_summary"
    assert crosswalk["blood test report".casefold()].id == "lab_results"


def test_sensitivity_ordering():
    taxonomy = default_taxonomy()
    assert taxonomy.max_sensitivity([]) == "none"
    assert taxonomy.max_sensitivity(["country"]) == "low"
    assert taxonomy.max_sensitivity(["country", "ssn"]) == "critical"
    assert taxonomy.max_sensitivity(["city", "email"]) == "high"


def test_phi_and_special_category_flags():
    taxonomy = default_taxonomy()
    assert taxonomy.pii("medical_record_number").phi
    assert taxonomy.pii("religious_belief").special_category
    assert not taxonomy.pii("company_name").phi


def test_unknown_lookups_raise():
    taxonomy = default_taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.pii("not_a_label")
    with pytest.raises(TaxonomyError):
        taxonomy.industry("agriculture")


def test_duplicate_subcategory_is_rejected(tmp_path):
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir()
    (taxonomy_dir / "pii.json").write_text(
        json.dumps(
            {
                "version": "0",
                "groups": {"identity": "x"},
                "sensitivity_levels": {"low": "x"},
                "labels": [
                    {"id": "a", "group": "identity", "sensitivity": "low", "description": "x"}
                ],
            }
        )
    )
    (taxonomy_dir / "industries.json").write_text(
        json.dumps(
            {
                "version": "0",
                "industries": [
                    {
                        "id": "healthcare",
                        "label": "H",
                        "description": "d",
                        "categories": [
                            {
                                "id": "c1",
                                "label": "C1",
                                "subcategories": [{"id": "dup", "label": "x", "description": "y"}],
                            },
                            {
                                "id": "c2",
                                "label": "C2",
                                "subcategories": [{"id": "dup", "label": "x", "description": "y"}],
                            },
                        ],
                    }
                ],
            }
        )
    )
    with pytest.raises(TaxonomyError, match="duplicate subcategory"):
        load_taxonomy(taxonomy_dir)
