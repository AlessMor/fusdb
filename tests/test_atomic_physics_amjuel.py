from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from fusdb.registry import RelationRegistry
from fusdb.registry.dataset import load_dataset
from fusdb.utils.datasets import evaluate_amjuel_h2_rate, evaluate_amjuel_h4_rate


AMJUEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fusdb"
    / "registry"
    / "dataset"
    / "atomic_reactions"
)
FORTRAN_FLOAT = re.compile(r"^[+-]?\d+\.\d+(?:[DEde][+-]\d+)$")


def _yaml_paths() -> list[Path]:
    return sorted(AMJUEL_DIR.rglob("polynomialfit_AMJUEL-H2-*.yaml"))


def test_all_amjuel_h2_yaml_files_keep_expected_shape():
    paths = _yaml_paths()
    assert len(paths) == 54

    for path in paths:
        data = yaml.safe_load(path.read_text())
        assert data["schema_version"] == 1
        assert data["datatype"] == "polynomialfit"
        assert data["source"].startswith("AMJUEL-H2-")
        assert data["database"] == "AMJUEL"
        assert data["amjuel_section"] == "H.2"
        assert data["inputs"] == {"temperature": "T_edge"}
        assert data["output_unit"] == "m^3/s"
        assert data["source_output_unit"] == "cm^3/s"
        assert data["category"]
        assert data["species"]
        assert data["output"].endswith("_rate")
        assert data["relation_name"].startswith("AMJUEL H.2 ")
        assert data["formula"] == "ln(<sigma v>) = sum_i b_i * ln(T_eV)^i"

        coefficients = data["coefficients"]
        assert len(coefficients) == 9
        assert all(FORTRAN_FLOAT.match(value) for value in coefficients)

        if "temperature_limits" in data:
            limits = data["temperature_limits"]
            if "min_eV" in limits:
                assert float(limits["min_eV"]) > 0.0
            if "max_eV" in limits:
                assert float(limits["max_eV"]) > 0.0


def test_amjuel_h2_evaluator_uses_edge_temperature_and_cm3_to_m3_conversion():
    dataset_id = "polynomialfit_AMJUEL-H2-2.19_H-charge-exchange"

    # At T_edge = 1e-3 keV, AMJUEL has T = 1 eV and ln(T) = 0, so only b0 contributes.
    expected = math.exp(-1.850280000000e01) * 1.0e-6
    assert evaluate_amjuel_h2_rate(dataset_id, T_edge=1.0e-3) == pytest.approx(expected, rel=1.0e-12)

    values = evaluate_amjuel_h2_rate(dataset_id, T_edge=np.array([1.0e-3, 1.0e-2]))
    assert values.shape == (2,)
    assert np.all(values >= 0.0)


def test_amjuel_h2_constant_langevin_charge_exchange_fit():
    dataset_id = "polynomialfit_AMJUEL-H2-2.25_H-charge-exchange"

    low = evaluate_amjuel_h2_rate(dataset_id, T_edge=1.0e-3)
    high = evaluate_amjuel_h2_rate(dataset_id, T_edge=1.0)
    assert low == pytest.approx(2.0e-14, rel=5.0e-10)
    assert high == pytest.approx(low, rel=1.0e-12)


def test_amjuel_h2_relations_register_and_duplicate_defaults_are_selected():
    registry = RelationRegistry.discover()

    h_cx_producers = registry.producers("H_charge_exchange_rate")
    assert {rel.name for rel in h_cx_producers} == {
        "AMJUEL H.2 2.19 H charge exchange rate",
        "AMJUEL H.2 2.20 H charge exchange Freeman-Jones rate",
        "AMJUEL H.2 2.25 H charge exchange Langevin rate",
    }

    selected = registry.get_filtered_relations(variables=["H_charge_exchange_rate"])
    assert [rel.name for rel in selected if "H charge exchange" in rel.name] == [
        "AMJUEL H.2 2.25 H charge exchange Langevin rate"
    ]

    h2_selected = registry.get_filtered_relations(variables=["H2_charge_exchange_rate"])
    assert [rel.name for rel in h2_selected if "H2 charge exchange" in rel.name] == [
        "AMJUEL H.2 2.26 H2 charge exchange rate"
    ]

    he_selected = registry.get_filtered_relations(variables=["He_ionization_rate"])
    assert [rel.name for rel in he_selected if "He ionization" in rel.name] == [
        "AMJUEL H.2 2.36 He ionization STRAHL rate"
    ]

    rel = registry.get("AMJUEL H.2 2.25 H charge exchange Langevin rate")
    assert rel.outputs == ("H_charge_exchange_rate",)
    assert "Source: AMJUEL H.2 coefficient fit." in (rel.func.__doc__ or "")


def test_amjuel_h2_relations_have_temperature_validity_constraints():
    registry = RelationRegistry.discover()
    amjuel_relations = [rel for rel in registry if rel.name.startswith("AMJUEL H.2 ")]
    assert len(amjuel_relations) == 54

    for rel in amjuel_relations:
        assert set(rel.input_names) == {"T_edge"}
        constraint_text = set(rel.constraints)
        assert "T_edge > 0.0" in constraint_text
        assert not any("n_e_edge" in text for text in constraint_text)
        assert rel.constraint_relations


def test_amjuel_h2_relation_includes_temperature_limit_when_amjuel_reports_tmin():
    registry = RelationRegistry.discover()
    rel = registry.get("AMJUEL H.2 2.21 H2 dissociation original rate")

    assert "T_edge >= 0.00126" in rel.constraints


def _h4_yaml_paths() -> list[Path]:
    return sorted(AMJUEL_DIR.rglob("polynomialfit_AMJUEL-H4-*.yaml"))


def test_all_amjuel_h4_yaml_files_keep_expected_shape():
    paths = _h4_yaml_paths()
    assert len(paths) == 8

    for path in paths:
        data = yaml.safe_load(path.read_text())
        assert data["schema_version"] == 1
        assert data["datatype"] == "polynomialfit"
        assert data["source"].startswith("AMJUEL-H4-")
        assert data["database"] == "AMJUEL"
        assert data["amjuel_section"] == "H.4"
        if "inputs" in data:
            assert data["inputs"] == {"density": "n_e_edge", "temperature": "T_edge"}
        if "output_unit" in data:
            assert data["output_unit"] == "m^3/s"
            assert data["source_output_unit"] == "cm^3/s"
        assert data.get("relation_name", data.get("name", "")).startswith("AMJUEL H.4 ")
        assert data["output"].endswith("_rate")
        assert float(str(data["density_limits"]["min_cm3"]).replace("D", "E")) > 0.0
        assert float(str(data["density_limits"]["max_cm3"]).replace("D", "E")) > 0.0

        blocks = data["coefficient_blocks"]
        covered_density_indices = sorted(
            index for block in blocks for index in block["density_indices"]
        )
        assert covered_density_indices == list(range(9))
        for block in blocks:
            rows = block["rows"]
            assert [row["temperature_index"] for row in rows] == list(range(9))
            for row in rows:
                assert len(row["coefficients"]) == len(block["density_indices"])
                assert all(math.isfinite(float(str(value).replace("D", "E"))) for value in row["coefficients"])


def test_amjuel_h4_evaluator_uses_edge_units_and_density_scaling():
    dataset_id = "polynomialfit_AMJUEL-H4-2.1.8_H-recombination"

    # At T_edge = 1e-3 keV (1 eV) and n_e_edge = 1e14 m^-3 (1e8 cm^-3) both fit
    # logarithms vanish, so only the (0, 0) coefficient contributes.
    data = load_dataset(dataset_id).data
    a00 = float(data["coefficient_blocks"][0]["rows"][0]["coefficients"][0].replace("D", "E"))
    expected = math.exp(a00) * 1.0e-6
    assert evaluate_amjuel_h4_rate(dataset_id, n_e_edge=1.0e14, T_edge=1.0e-3) == pytest.approx(expected, rel=1.0e-12)

    values = evaluate_amjuel_h4_rate(dataset_id, n_e_edge=np.array([1.0e14, 1.0e15]), T_edge=1.0e-3)
    assert values.shape == (2,)
    assert np.all(values >= 0.0)


def test_amjuel_h4_evaluator_clips_density_to_fit_limits():
    dataset_id = "polynomialfit_AMJUEL-H4-2.1.8_H-recombination"

    below_limit = evaluate_amjuel_h4_rate(dataset_id, n_e_edge=1.0e12, T_edge=1.0e-2)
    at_limit = evaluate_amjuel_h4_rate(dataset_id, n_e_edge=1.0e14, T_edge=1.0e-2)
    assert below_limit == pytest.approx(at_limit, rel=1.0e-12)


@pytest.mark.parametrize(
    ("dataset_id", "a00"),
    (
        ("polynomialfit_AMJUEL-H4-3.2.3r_H2-mar-via-h2-plus", -2.191302446846e01),
        ("polynomialfit_AMJUEL-H4-3.2.3d_H2-mad-via-h2-plus", -2.305748927979e01),
        ("polynomialfit_AMJUEL-H4-3.2.3i_H2-mai-via-h2-plus", -4.373131541734e01),
        ("polynomialfit_AMJUEL-H4-2.2.17r_H2-mar-via-h-minus", -2.297800283146e01),
        ("polynomialfit_AMJUEL-H4-2.2.17d_H2-mad-via-h-minus", -3.882083547683e01),
    ),
)
def test_amjuel_condensed_molecular_fits_match_their_corona_limit(dataset_id, a00):
    # At 1 eV and 1e14 m^-3, both AMJUEL fit logarithms are zero. The rate is
    # therefore exactly exp(a00) cm^3/s, converted to m^3/s.
    expected = math.exp(a00) * 1.0e-6
    assert evaluate_amjuel_h4_rate(dataset_id, 1.0e14, 1.0e-3) == pytest.approx(
        expected, rel=1.0e-12
    )

    values = evaluate_amjuel_h4_rate(
        dataset_id,
        np.logspace(14, 22, 9),
        1.0e-3,
    )
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_amjuel_h4_relations_register_with_density_constraints():
    registry = RelationRegistry.discover()
    h4_relations = [rel for rel in registry if rel.name.startswith("AMJUEL H.4 ")]
    assert len(h4_relations) == 8

    for rel in h4_relations:
        assert set(rel.input_names) == {"n_e_edge", "T_edge"}
        constraint_text = set(rel.constraints)
        assert "n_e_edge >= 1e+14" in constraint_text
        assert "n_e_edge <= 1e+22" in constraint_text
        assert "T_edge > 0.0" in constraint_text

    rel = registry.get("AMJUEL H.4 2.1.8 H recombination rate")
    assert rel.outputs == ("H_recombination_rate",)
    assert "T_edge >= 0.0001" in rel.constraints
    assert "Source: AMJUEL H.4 coefficient fit." in (rel.func.__doc__ or "")
