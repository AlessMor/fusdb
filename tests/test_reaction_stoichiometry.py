"""The hand-written reaction coefficients must agree with reactions.yaml.

``plasma_balance_ode`` and ``neutron_production_rate`` transcribe stoichiometry
by hand (they are on the batched-popcon hot path).  These tests are what keep
``registry/reactions.yaml`` honest: they fail if the two ever diverge.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from fusdb.registry import REACTIONS, SPECIES
from fusdb.relations.composition.densities import _reaction_balances
from fusdb.relations.fusion_reactions import neutronics
from fusdb.relations.fusion_reactions.power import aggregate

BALANCE_SPECIES = ("D", "T", "He3", "He4", "p")
DENSITIES = {"D": 3.0e20, "T": 7.0e19, "He3": 1.3e18, "He4": 5.5e18, "p": 2.1e18}
SIGMAV_ARGS = (
    "sigmav_DT", "sigmav_DDn", "sigmav_DDp", "sigmav_DHe3", "sigmav_TT",
    "sigmav_He3He3", "sigmav_THe3_D", "sigmav_THe3_np",
)


def _expected_rate(spec, sigmav: float) -> float:
    a, b = spec.reactants
    rate = DENSITIES[a] * DENSITIES[b] * sigmav
    return 0.5 * rate if spec.like_reactants else rate


@pytest.mark.parametrize("key", sorted(REACTIONS))
def test_balance_ode_matches_registry_stoichiometry(key: str) -> None:
    spec = REACTIONS[key]
    sigmav = 1.0e-22
    kwargs = {name: 0.0 for name in SIGMAV_ARGS}
    kwargs[f"sigmav_{key}"] = sigmav
    balances = _reaction_balances(
        DENSITIES["D"], DENSITIES["T"], DENSITIES["He3"], DENSITIES["He4"], **kwargs
    )
    rate = _expected_rate(spec, sigmav)
    for species, actual in zip(BALANCE_SPECIES, balances):
        expected = spec.stoichiometry(species) * rate
        assert actual == pytest.approx(expected, rel=1e-12, abs=0.0), (
            f"{key}: d{species}/dt coefficient is {actual / rate if rate else actual}, "
            f"reactions.yaml says {spec.stoichiometry(species)}"
        )


def test_neutron_rate_matches_registry_multiplicities() -> None:
    signature = inspect.signature(neutronics.neutron_production_rate.func)
    args = {name: 0.0 for name in signature.parameters}
    for spec in REACTIONS.values():
        yield_n = spec.stoichiometry("n")
        argument = f"Rr_{spec.key}"
        if yield_n <= 0:
            assert argument not in args, f"{spec.key} produces no neutrons but is summed into neutron_rate"
            continue
        assert argument in args, (
            f"{spec.key} yields {yield_n} neutron(s) in reactions.yaml but is missing from neutron_production_rate"
        )
        one = dict(args, **{argument: 1.0})
        assert neutronics.neutron_production_rate.func(**one) == pytest.approx(float(yield_n))


def test_charged_and_neutron_power_partition_every_product() -> None:
    charged = set(inspect.signature(aggregate.charged_fusion_power.func).parameters)
    neutral = set(inspect.signature(aggregate.neutron_fusion_power.func).parameters)
    assert not (charged & neutral), "a channel is summed into both P_charged and P_neutron"
    for spec in REACTIONS.values():
        for product in set(spec.products):
            if spec.energies_j.get(product) is None:
                continue
            target = neutral if product == "n" else charged
            assert any(name.startswith(f"P_fus_{spec.key}") for name in target), (
                f"{spec.key} product {product!r} is not aggregated into fusion power"
            )


def test_registry_species_are_known() -> None:
    for spec in REACTIONS.values():
        for species in (*spec.reactants, *spec.products):
            assert species in SPECIES, f"{spec.key} references unknown species {species!r}"
