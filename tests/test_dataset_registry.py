from __future__ import annotations

from collections import Counter

import numpy as np

from fusdb.registry.dataset import DATASETS, dataset_filename, load_dataset, load_table
from fusdb.relations.radiation.impurity_radiation import (
    _mavrin_coronal_Lz,
    _process_coronal_Lz,
)
from fusdb.utils.datasets import (
    evaluate_amjuel_h2_rate,
    reactivity_from_reactivity_table,
    reactivity_from_xsection_table,
)


def test_all_packaged_datasets_use_common_envelope_and_canonical_name():
    datatype_counts: Counter[str] = Counter()
    for dataset_id in DATASETS:
        document = load_dataset(dataset_id)
        assert document.data["schema_version"] == 1
        assert document.path.name == dataset_filename(
            document.datatype, document.source, document.subject
        )
        assert document.dataset_id == dataset_id
        datatype_counts[document.datatype] += 1

    assert datatype_counts == {
        "xsection": 16,
        "reactivity": 5,
        "polynomialfit": 87,
        "coolingcurve": 21,  # 11 PROCESS + 10 radas coronal (no radas Kr: all-zero source table)
        "meancharge": 11,  # radas coronal mean charge: He Li Be C N O Ne Ar Kr Xe W
    }


def test_radiation_resources_are_split_per_species():
    for symbol in ("He", "C", "Ar", "W"):
        coronal = load_dataset(f"polynomialfit_mavrin_coronal_{symbol}")
        process = load_dataset(f"coolingcurve_PROCESS_coronal_{symbol}")
        radas = load_dataset(f"coolingcurve_radas_coronal_{symbol}")
        assert coronal.subject == symbol
        assert "radc" in coronal.data
        for table in (process, radas):
            assert table.subject == symbol
            assert len(table.data["temperature_keV"]) == len(table.data["Lz_Wm3"])


def test_atomic_resources_mirror_the_relation_hierarchy():
    ionization = load_dataset("polynomialfit_AMJUEL-H2-2.17_H-ionization")
    mar = load_dataset("polynomialfit_AMJUEL-H4-3.2.3r_H2-mar-via-h2-plus")
    assert ionization.path.parent.name == "ionization"
    assert mar.path.parent.name == "via_h2_plus"
    assert mar.path.parent.parent.name == "mar"


def test_split_radiation_datasets_feed_both_cooling_curve_evaluators():
    temperatures = np.asarray([0.1, 1.0, 10.0])
    for evaluator in (_mavrin_coronal_Lz, _process_coronal_Lz):
        values = evaluator("C", temperatures)
        assert values.shape == temperatures.shape
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)


def test_table_parser_and_dataset_operations_share_registry_ids():
    table = load_table("xsection_ENDFB-VIII0_DDn")
    assert table.quantities == ("energy", "cross_section")
    assert table.units == ("ev", "barn")
    assert all(column.ndim == 1 for column in table.columns)

    xsection_rate = reactivity_from_xsection_table("xsection_ENDFB-VIII0_DDn", 10.0)
    direct_rate = reactivity_from_reactivity_table("reactivity_NRL_DT", 10.0)
    assert np.isfinite(xsection_rate) and xsection_rate > 0.0
    assert np.isfinite(direct_rate) and direct_rate > 0.0


def test_atomic_fit_operation_accepts_stable_dataset_id():
    value = evaluate_amjuel_h2_rate(
        "polynomialfit_AMJUEL-H2-2.29_N2-molecular-ionization", 1.0e-2
    )
    assert np.isfinite(value) and value > 0.0
