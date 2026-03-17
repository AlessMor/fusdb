from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.registry import load_allowed_species
from fusdb.relations.reactivities.tabulated_reactivities import load_table
from fusdb.relations.reactivities.tabulated_reactivities import prepare_table
from fusdb.relations.reactivities.tabulated_reactivities import reactivity_from_reactivity_table
from fusdb.relations.reactivities.tabulated_reactivities import reactivity_from_xsection_table


TABLES_DIR = ROOT / "src" / "fusdb" / "relations" / "reactivities" / "reactivity_tables"


def test_all_yaml_tables_use_yaml_metadata_and_parse_as_numeric_columns():
    """Expected: every reactivity table uses the YAML metadata plus CSV data-block schema."""
    table_paths = sorted(TABLES_DIR.glob("*.yaml"))

    assert table_paths
    for path in table_paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert raw["table_kind"] in {"cross_section", "reactivity"}
        assert isinstance(raw.get("columns"), list)
        assert len(raw["columns"]) == 2
        assert isinstance(raw.get("data"), str)
        assert raw["data"].strip()

        table = load_table(path)
        assert len(table.quantities) == 2
        assert len(table.units) == 2
        assert len(table.columns) == 2
        assert table.columns[0].size > 0
        assert table.columns[1].size > 0
        if raw["table_kind"] == "cross_section":
            assert table.units == ("ev", "barn")
        else:
            assert table.quantities == ("temperature", "sigmav")
            assert table.units == ("ev", "cm^3/s")


def test_unregistered_viii1_yaml_tables_can_be_read_by_filename():
    """Expected: the generic reader can parse any YAML table in reactivity_tables and select metadata by key."""
    table = load_table(
        "DT_xsection_ENDFB-VIII1.yaml",
        metadata_keys=("library", "reaction", "reactants", "reference_frame", "mt", "missing"),
    )

    assert table.path == TABLES_DIR / "DT_xsection_ENDFB-VIII1.yaml"
    assert table.quantities == ("energy", "cross_section")
    assert table.units == ("ev", "barn")
    assert table.metadata == {
        "library": "ENDF/B-VIII.1",
        "reaction": "H-3(D,N`)HE-4-L0,SIG",
        "reactants": {"projectile": "D", "target": "T"},
        "reference_frame": "lab",
        "mt": 50,
    }
    assert table.columns[0][0] == 100.0
    assert table.columns[1][0] == 2.0469e-56


def test_reactivity_tables_convert_header_units_to_si():
    """Expected: direct sigmav tables use eV on disk and convert to keV/m^3/s in memory."""
    raw_table = load_table("DT_reactivity_NRL.yaml")
    table = prepare_table(
        "DT_reactivity_NRL.yaml",
        expected_kind="reactivity",
        quantities=("temperature", "sigmav"),
        units=("ev", "cm^3/s"),
        scales=(1.0e-3, 1.0e-6),
        scaled_units=("kev", "m^3/s"),
        positive_columns=(0, 1),
        sort_by=0,
        unique_by=0,
    )

    assert raw_table.units == ("ev", "cm^3/s")
    assert np.isclose(raw_table.columns[0][0], 1000.0)
    assert table.units == ("kev", "m^3/s")
    assert np.isclose(table.columns[0][0], 1.0)
    assert np.isclose(table.columns[1][0], 5.5e-27)
    assert np.all(table.columns[0] > 0.0)
    assert np.all(table.columns[1] > 0.0)


def test_cross_section_tables_respect_header_units_before_integration():
    """Expected: cross-section loaders consume the uniform eV/barn table format."""
    dhe3_raw = load_table("DHe3_xsection_ENDFB-VIII0.yaml")
    dt_table = prepare_table(
        "DT_xsection_ENDFB-VIII0.yaml",
        expected_kind="cross_section",
        quantities=("energy", "cross_section"),
        units=("ev", "barn"),
        scales=(1.0e-3, 1.0e-28),
        scaled_units=("kev", "m^2"),
        sort_by=0,
        unique_by=0,
    )
    dhe3_table = prepare_table(
        "DHe3_xsection_ENDFB-VIII0.yaml",
        expected_kind="cross_section",
        quantities=("energy", "cross_section"),
        units=("ev", "barn"),
        scales=(1.0e-3, 1.0e-28),
        scaled_units=("kev", "m^2"),
        sort_by=0,
        unique_by=0,
    )

    assert dhe3_raw.units == ("ev", "barn")
    assert np.isclose(dhe3_raw.columns[0][0], 1187.5)
    assert dt_table.units == ("kev", "m^2")
    assert dhe3_table.units == ("kev", "m^2")
    assert dt_table.columns[0].shape == dt_table.columns[1].shape
    assert dhe3_table.columns[0].shape == dhe3_table.columns[1].shape
    assert np.any(dt_table.columns[1] > 0.0)
    assert np.any(dhe3_table.columns[1] > 0.0)


def test_explicit_table_helpers_cover_direct_and_cross_section_tables():
    """Expected: public helpers expose direct-rate and cross-section-backed table evaluation."""
    temperature_keV = np.asarray([3.0, 30.0, 300.0], dtype=float)

    dt_nrl = np.asarray(
        reactivity_from_reactivity_table("DT_reactivity_NRL.yaml", temperature_keV, interpolation_kind="linear"),
        dtype=float,
    )
    dt_endfb = np.asarray(
        reactivity_from_xsection_table("DT_xsection_ENDFB-VIII0.yaml", temperature_keV),
        dtype=float,
    )

    assert dt_nrl.shape == temperature_keV.shape
    assert dt_endfb.shape == temperature_keV.shape
    assert np.all(dt_nrl > 0.0)
    assert np.all(dt_endfb > 0.0)


def test_cross_section_helper_supports_lab_and_cm_reference_frames():
    """Expected: the x-section helper lets callers opt out of the lab-to-CM conversion."""
    temperature_keV = np.asarray([3.0, 30.0, 300.0], dtype=float)

    dt_lab = np.asarray(
        reactivity_from_xsection_table(
            "DT_xsection_ENDFB-VIII0.yaml",
            temperature_keV,
            reference_frame="lab",
        ),
        dtype=float,
    )
    dt_cm = np.asarray(
        reactivity_from_xsection_table(
            "DT_xsection_ENDFB-VIII0.yaml",
            temperature_keV,
            reference_frame="cm",
        ),
        dtype=float,
    )

    assert np.all(dt_lab > 0.0)
    assert np.all(dt_cm > 0.0)
    assert not np.allclose(dt_lab, dt_cm, rtol=1.0e-4, atol=0.0)


def test_cm_reference_frame_matches_preconverted_cm_table(tmp_path: Path):
    """Expected: a CM-table file reproduces the lab-table result when its metadata says ``reference_frame: cm``."""
    temperature_keV = np.asarray([3.0, 30.0, 300.0], dtype=float)
    species = load_allowed_species()
    mass_d = float(species["D"]["isotopic_mass_u"])
    mass_t = float(species["T"]["isotopic_mass_u"])
    lab_to_cm = mass_t / (mass_d + mass_t)

    source_path = TABLES_DIR / "DT_xsection_ENDFB-VIII0.yaml"
    target_path = tmp_path / "DT_xsection_CM-TEMP.yaml"
    source_document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    assert isinstance(source_document, dict)
    converted_rows = []
    for row in source_document["data"].splitlines():
        stripped = row.strip()
        if not stripped:
            continue
        energy_ev, cross_section_barn = [token.strip() for token in stripped.split(",", maxsplit=1)]
        converted_rows.append(f"{float(energy_ev) * lab_to_cm:.16g},{cross_section_barn}")
    source_document["reference_frame"] = "cm"
    source_document["data"] = "\n".join(converted_rows)
    target_path.write_text(yaml.safe_dump(source_document, sort_keys=False), encoding="utf-8")

    dt_lab = np.asarray(
        reactivity_from_xsection_table("DT_xsection_ENDFB-VIII0.yaml", temperature_keV),
        dtype=float,
    )
    dt_cm = np.asarray(
        reactivity_from_xsection_table(target_path, temperature_keV),
        dtype=float,
    )

    assert np.allclose(dt_lab, dt_cm, rtol=1.0e-12, atol=0.0)


def test_cross_section_helper_rejects_unknown_reference_frame():
    """Expected: the x-section helper validates the requested reference frame."""
    with pytest.raises(ValueError, match="Unsupported reference_frame"):
        reactivity_from_xsection_table(
            "DT_xsection_ENDFB-VIII0.yaml",
            np.asarray([10.0], dtype=float),
            reference_frame="plasma",
        )


def test_reactivity_helper_requires_explicit_table_filename():
    """Expected: direct-table helpers no longer rewrite shorthand table names."""
    with pytest.raises(FileNotFoundError):
        reactivity_from_reactivity_table(
            "THe3_total_reactivity.yaml",
            np.asarray([3.0, 30.0, 300.0], dtype=float),
            interpolation_kind="linear",
        )


def test_generic_table_preparation_rejects_legacy_reaction_ids():
    """Expected: generic table preparation accepts explicit filenames, not legacy reaction ids."""
    with pytest.raises(FileNotFoundError):
        prepare_table(
            "DT_NRL",
            expected_kind="reactivity",
            quantities=("temperature", "sigmav"),
            units=("ev", "cm^3/s"),
            scales=(1.0e-3, 1.0e-6),
        )

    with pytest.raises(FileNotFoundError):
        prepare_table(
            "DT_ENDFB_VIII0",
            expected_kind="cross_section",
            quantities=("energy", "cross_section"),
            units=("ev", "barn"),
            scales=(1.0e-3, 1.0e-28),
        )
