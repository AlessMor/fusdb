import numpy as np

from fusdb.registry import RELATIONS, TAGS, VARIABLES
from fusdb.profiles.numerics import volume_average


def test_rho_is_computational_and_legacy_minor_radius_aliases_resolve_to_rho_minor():
    assert VARIABLES.resolve("rho") == "rho"
    assert VARIABLES.resolve("normalized_minor_radius") == "rho_minor"
    assert VARIABLES.resolve("r_over_a") == "rho_minor"
    assert "computational" in VARIABLES.get("rho").description.lower()


def test_tokamak_default_coordinate_relations_are_selected():
    names = {
        relation.name
        for relation in RELATIONS.get_filtered_relations(tags=TAGS.expand(("tokamak",)))
    }
    assert "Tokamak normalized minor-radius coordinate" in names
    assert "Tokamak normalized enclosed volume" in names
    assert "Tokamak volume integration weight" in names


def test_default_tokamak_mappings_preserve_legacy_coordinate_and_volume_weight():
    rho = np.linspace(0.0, 1.0, 46)

    rho_minor = RELATIONS.get("Tokamak normalized minor-radius coordinate").solve({"rho": rho})
    v_norm = RELATIONS.get("Tokamak normalized enclosed volume").solve({"rho": rho})
    w_v = RELATIONS.get("Tokamak volume integration weight").solve({"rho": rho})

    np.testing.assert_allclose(rho_minor, rho)
    np.testing.assert_allclose(v_norm, rho**2)
    np.testing.assert_allclose(w_v, rho)

    profile = 1.0 + 2.0 * rho + rho**2
    legacy = volume_average(profile, rho)
    explicit = volume_average(profile, rho, weight=w_v)
    np.testing.assert_allclose(explicit, legacy, rtol=0.0, atol=1.0e-14)


def test_volume_average_consistency_uses_explicit_weight_without_changing_tokamak_result():
    rho = np.linspace(0.0, 1.0, 46)
    profile = 3.0 - rho**2
    average = volume_average(profile, rho)
    relation = RELATIONS.get("Electron density volume-average consistency")

    status = relation.verify_status(
        {
            "n_e_avg": average,
            "n_e": profile,
            "w_V": rho,
            "rho": rho,
        }
    )
    assert status["verified"]


def test_physical_line_average_uses_rho_minor_and_is_tokamak_scoped():
    relation = RELATIONS.get("Electron density line-average")
    assert relation.input_names == ("n_e", "rho_minor")
    assert "tokamak" in relation.tags
    assert "stellarator" not in relation.tags
    assert "mirror" not in relation.tags

    rho_minor = np.linspace(0.0, 1.0, 46)
    profile = 2.0 + rho_minor
    result = relation.solve({"n_e": profile, "rho_minor": rho_minor})
    np.testing.assert_allclose(result, 2.5)
