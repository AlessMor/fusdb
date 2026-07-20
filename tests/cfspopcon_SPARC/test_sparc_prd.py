"""cfspopcon SPARC PRD reproduction: full-grid scan vs ``output/dataset.nc``.

This rewrites the earlier single-point ``PRD.json`` comparison into a full 40x30
operating-space comparison.  ``Reactor.popcon`` scans the *same*
``(average_electron_density, average_electron_temp)`` grid cfspopcon scanned in
``input.yaml`` (the grid block: n_e 1..40 x1e19, T_e 5..20 keV), and every field
is compared cell-by-cell against cfspopcon's reference ``output/dataset.nc``
(read with ``h5py``; see :mod:`_compare`).

The fixture (``reactor.yaml``) is driven from input.yaml's inputs: geometry,
field, current, and the impurity concentrations (He 6e-2, O 3.1e-3, W 1.5e-5
of n_e), which drive the Mavrin-coronal line radiation, the derived ``Z_eff``
(~1.34, vs the radas-based reference within ~1%) and the scalar ion-inventory
anchor ``n_i_avg`` through the Mavrin 2018 T-dependent mean charges (the
f-fractions keep the species split: He as ``f_He4``, O+W folded into the
generic impurity).  Radiation runs on cfspopcon's composition (hydrogenic
bremsstrahlung + synchrotron + line radiation) with the radas coronal Lz
tables -- the reference's own radiation method.  Profiles carry the
electron AND ion Angioni density peakings (the pointwise ``n_i ~ n_e`` tie is
a weak default; the ion profile decouples through its own generator), plus
the Martin+Ryter L-H scaling and cfspopcon's stored-energy / ITER98y2
conventions, and the SOL two-point model on input.yaml's fixed-target-Te
block.  Field-by-field agreement (median relative error over
the ~1029 certified cells), measured and categorised in :data:`_compare.FIELDS`:

  * **match** -- fusdb reproduces cfspopcon within ~10% grid-wide: geometry,
    the confinement solve (``P_in`` ~0.2%, ``tau_E`` ~0.1% on the cfspopcon
    W/ITER98y2 conventions), both Angioni peakings (~0.4%), the whole
    resistive/collisionality chain (~0.4%), the Barr inductance/flux chain,
    ``P_ohmic`` (~3%), ``P_radiation``/``f_rad`` (~3%, on the radas Lz
    tables), ``P_sep`` (~0.3%), ``P_SOL/P_LH`` (~8.5%, inheriting ``P_LH``'s
    systematic), ``P_aux`` (~2%), ``P_LH`` (~9%), the triple product / peak
    fuel density (~7%, total-vs-fuel-ion semantics), and ``Q`` itself (~9% on
    cfspopcon's own definition).
  * **fusion** -- ``P_fus``/``P_neutron``/``P_alpha`` land at ~6% median
    (PRD cell +0.8%): fuel densities and profile shapes both match
    cfspopcon's; the residual is reactivity-table differences plus the
    non-DT channels cfspopcon does not compute.
  * **gap** -- known modelling differences carried as ``xfail`` so a future fix
    surfaces as an ``xpass``: the beta/pressure chain (~13%, cfspopcon's
    dilution-free product-of-averages pressure), bootstrap (~12%) and the CS
    flux (~14%).
  * **missing** -- fields the scan fixture does not compute (the Lengyel edge
    seeding, flat-top duration); guarded so the set is explicit.

The ``comparison.ipynb`` notebook in this directory renders the same comparison
as a full table (including the missing fields) plus the T-n contour overlay.
"""

from __future__ import annotations

import numpy as np
import pytest

from _compare import FIELDS, SCAN_OUTPUTS, compare, load_dataset, run_scan
from fusdb.registry import VARIABLES

MATCH_TOL = 0.10   # match fields: median relative error over the certified grid
FUSION_TOL = 0.10  # fusion chain: ~6% median (fuel profile on its own ion peaking)


@pytest.fixture(scope="module")
def grid_comparison():
    """Run the scan once, load the reference, compare every field; cache per module."""
    pytest.importorskip("h5py", reason="reference dataset.nc is netCDF4/HDF5")
    result = run_scan(outputs=SCAN_OUTPUTS)
    assert result["success"], result.get("errors")
    with load_dataset() as handle:
        comparisons = compare(result, handle)
    return result, {c.dataset_name: c for c in comparisons}


def test_scan_certifies_operating_region(grid_comparison):
    """The batched scan certifies the bulk of the 40x30 grid (infeasible corners
    -- ignited, sub-L-H, or out-of-reactivity-range -- are legitimately masked)."""
    result, _ = grid_comparison
    payload = result["popcon"]
    assert payload["success"].shape == (30, 40)
    assert payload["success"].sum() >= 700  # ~1029 today (Angioni-peaked h_mode; the cold/ignited fringe and the n_e=1e19 column fail)


_MATCH = [d for d, _f, cat in FIELDS if cat == "match"]
_FUSION = [d for d, _f, cat in FIELDS if cat == "fusion"]
_GAP = [d for d, _f, cat in FIELDS if cat == "gap"]
_MISSING = [d for d, _f, cat in FIELDS if cat == "missing"]


@pytest.mark.parametrize("dataset_name", _MATCH)
def test_match_field_within_tolerance(grid_comparison, dataset_name):
    """Fields fusdb genuinely reproduces agree with cfspopcon within 10% (median)."""
    _, by_name = grid_comparison
    c = by_name[dataset_name]
    assert c.computed, f"{dataset_name}: scan produced no comparable cells"
    assert c.median_rel <= MATCH_TOL, (
        f"{dataset_name} ({c.fusdb_name}): median rel error "
        f"{c.median_rel:.1%} exceeds {MATCH_TOL:.0%} over {c.n_cells} certified cells"
    )


@pytest.mark.parametrize("dataset_name", _FUSION)
def test_fusion_field_within_tolerance(grid_comparison, dataset_name):
    """The fusion chain reproduces cfspopcon at ~10% median across the grid."""
    _, by_name = grid_comparison
    c = by_name[dataset_name]
    assert c.computed, f"{dataset_name}: scan produced no comparable cells"
    assert c.median_rel <= FUSION_TOL, (
        f"{dataset_name} ({c.fusdb_name}): median rel error {c.median_rel:.1%} "
        f"exceeds {FUSION_TOL:.0%}"
    )


@pytest.mark.parametrize("dataset_name", _GAP)
@pytest.mark.xfail(strict=False, reason="known fusdb<->cfspopcon modelling gap (see _compare.FIELDS)")
def test_gap_field_diverges(grid_comparison, dataset_name):
    """Documented modelling gaps: computed but outside 10%.

    Carried as a (non-strict) xfail rather than dropped, so a future modelling
    fix that brings the field within 10% turns this into an xpass and flags that
    the field should be promoted to ``match``.
    """
    _, by_name = grid_comparison
    c = by_name[dataset_name]
    assert c.computed, f"{dataset_name}: scan produced no comparable cells"
    assert c.median_rel <= MATCH_TOL, (
        f"{dataset_name} ({c.fusdb_name}): median rel error {c.median_rel:.1%}"
    )


@pytest.mark.parametrize("dataset_name", _MISSING)
def test_missing_field_not_yet_computed(grid_comparison, dataset_name):
    """Fields the scan fixture does not compute stay explicitly documented.

    If fusdb starts producing one of these on the scan, this test fails on the
    now-comparable field -- the prompt to move it into ``FIELDS`` as match/gap.
    """
    _, by_name = grid_comparison
    c = by_name[dataset_name]
    assert not c.computed, (
        f"{dataset_name} ({c.fusdb_name}) is now computed on the scan "
        f"(median {c.median_rel:.1%}); reclassify it in _compare.FIELDS"
    )


def test_prd_point_matches_reference(grid_comparison):
    """Anchor check at the optimised PRD operating point (grid cell n_e=25e19,
    T_e=9.138 keV), the single point the old PRD.json comparison used."""
    result, _ = grid_comparison
    payload = result["popcon"]
    ix = int(np.argmin(np.abs(payload["x"]["values"] - 25.0e19)))
    iy = int(np.argmin(np.abs(payload["y"]["values"] - 9.13793)))
    assert payload["success"][iy, ix], "PRD operating point failed certification"
    with load_dataset() as handle:
        from _compare import dataset_grid

        # With the decoupled ion peaking the PRD cell is essentially exact:
        # P_in +0.2%, tau_E -0.1%, both peakings -0.4%, P_fus +0.8%.
        for fusdb_name, dataset_name, tol in [
            ("P_in", "P_in", 0.05),
            ("tau_E", "energy_confinement_time", 0.06),
            ("P_sep", "power_crossing_separatrix", 0.08),
            ("f_GW", "greenwald_fraction", 0.01),
            ("P_LH", "P_LH_thresh", 0.11),
            ("P_fus", "P_fusion", 0.05),
            ("density_peaking", "electron_density_peaking", 0.03),
            ("ion_density_peaking", "ion_density_peaking", 0.03),
        ]:
            got = payload["fields"][VARIABLES.resolve(fusdb_name)][iy, ix]
            ref = dataset_grid(handle, dataset_name)[iy, ix]
            assert got == pytest.approx(ref, rel=tol), f"{dataset_name}: {got:.6g} vs {ref:.6g}"
