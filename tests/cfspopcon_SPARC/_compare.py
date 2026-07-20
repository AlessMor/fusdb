"""Shared machinery for the cfspopcon SPARC full-grid comparison.

Both ``test_sparc_prd.py`` (the pytest regression guard) and
``comparison.ipynb`` (the human-readable diagnostic) drive off this module so
they can never drift apart.

The comparison runs fusdb's :meth:`Reactor.popcon` scan over the *same* 40x30
``(average_electron_density, average_electron_temp)`` grid cfspopcon scanned in
``input.yaml``, reads cfspopcon's reference ``output/dataset.nc``, converts every
reference field from its cfspopcon (pint) unit to the fusdb canonical/display
unit, and compares the two grids cell-by-cell over the certified operating
region.

The reference is a netCDF4/HDF5 file, read with ``h5py`` (fusdb ships no xarray).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fusdb.reactor import Reactor
from fusdb.registry import VARIABLES

CASE_DIR = Path(__file__).parent
REACTOR_YAML = CASE_DIR / "reactor.yaml"
DATASET_PATH = CASE_DIR / "output" / "dataset.nc"

# The scan grid is taken verbatim from input.yaml's ``grid`` block: electron
# density 1..40 (x1e19) x electron temperature 5..20 keV, linear, 40 x 30.
GRID_X = {"variable": "average_electron_density", "start": 1.0e19, "stop": 40.0e19, "num": 40}
GRID_Y = {"variable": "average_electron_temp", "start": 5.0, "stop": 20.0, "num": 30}
NX, NY = 40, 30

# cfspopcon pint unit string -> multiplicative factor onto the fusdb canonical
# (public) unit of the matching variable.  keV, T, m, s, Pa, A, V, Wb, H are
# shared, so those map to 1.0; power is W in fusdb (MW in cfspopcon), density is
# m^-3 (1e19/1e20 in cfspopcon), etc.  The two composite diagnostics
# (``PBpRnSq``, ``normalized_beta``) carry the awkward compound units cfspopcon
# builds them in; their factors are the pure unit conversion (1e6/1e40 and
# percent->fraction respectively), verified against fusdb at the PRD point.
UNIT_SCALE: dict[str, float] = {
    "meter": 1.0,
    "meter ** 2": 1.0,
    "meter ** 3": 1.0,
    "second": 1.0,
    "pascal": 1.0,
    "dimensionless": 1.0,
    "kiloelectron_volt": 1.0,
    "electron_volt": 1.0e-3,
    "megawatt": 1.0e6,
    "gigawatt": 1.0e9,
    "megajoule": 1.0e6,
    "_1e19_per_cubic_metre": 1.0e19,
    "_1e20_per_cubic_metre": 1.0e20,
    "_1e20_per_cubic_metre * kiloelectron_volt * second": 1.0e20,
    "ampere": 1.0,
    "volt": 1.0,
    "tesla": 1.0,
    "meter * ohm": 1.0,
    "weber": 1.0,
    "henry": 1.0,
    "millimeter": 1.0e-3,
    "1 / second": 1.0,
    "1 / meter ** 2 / second": 1.0,
    "megawatt / meter ** 2": 1.0e6,
    "gigawatt / meter ** 2": 1.0e9,
    "megawatt * tesla / meter": 1.0e6,
    "megawatt * tesla / _1e20_per_cubic_metre ** 2 / meter": 1.0e-34,
    "meter * percent * tesla / megaampere": 1.0e-2,
}


# Category of each mapped field, from the measured full-grid agreement
# (~1029/1200 certified cells, decoupled electron/ion Angioni peaking +
# cfspopcon W/tau_E/radiation conventions, Mavrin line radiation + derived
# Z_eff/n_i_avg from the c_z concentrations, He3 side-channels pruned,
# SOL two-point model on input.yaml's fixed-target-Te block, 2026-07-19):
#   "match"   fusdb reproduces cfspopcon within ~10% grid-wide  -> asserted
#   "fusion"  fusion chain: ~6% median (own ion peaking on fuel) -> asserted
#   "gap"     a known fusdb<->cfspopcon modelling difference      -> reported, xfail
#   "missing" fusdb does not (yet) compute this on the scan       -> reported only
#
# (dataset field, fusdb variable, category).  ``missing`` rows have their fusdb
# name as documentation of the intended target even though the scan yields NaN.
FIELDS: list[tuple[str, str, str]] = [
    # -- geometry / operating point --
    # fusdb's n_i_avg counts every ion species; the dataset's
    # average_ion_density is the *fuel* density (impurities separate), so the
    # 7.5% offset is the impurity inventory, not a solve error.
    ("average_ion_density", "n_i_avg", "match"),
    ("average_ion_temp", "T_i_avg", "match"),
    # Fuel-ion dilution n_fuel/n_e = 1 - sum(c_z Zbar_z) from the same Mavrin
    # mean charges as Z_eff/n_i_avg; ~0.006% grid-wide.
    ("dilution", "dilution", "match"),
    ("greenwald_fraction", "f_GW", "match"),
    ("peak_electron_temp", "T0", "match"),
    ("peak_ion_temp", "T_i_peak", "match"),
    # -- power / confinement balance --
    # P_in ~0.2% and tau_E ~0.1% grid-wide on the "Energy confinement time
    # ITER98y2 (cfspopcon)" + "Plasma stored energy (cfspopcon)" conventions.
    ("P_in", "P_in", "match"),
    ("energy_confinement_time", "tau_E", "match"),
    # Computed since "Plasma stored energy (cfspopcon)" landed (2026-07-17);
    # the scan reproduces the reference to ~0.002% median grid-wide.
    ("plasma_stored_energy", "W_th", "match"),
    # cfspopcon's diagnostic average_total_pressure is n_e(T_e+T_i) -- no
    # dilution, product of averages -- while fusdb's p_th integrates the real
    # peaked profiles with the real n_i; with the Angioni-peaked density the
    # two differ ~15% by construction, so the whole beta chain built on it is
    # a documented convention gap (importable as cfspopcon-form beta relations
    # if ever needed; the *energy* balance already uses cfspopcon's W).
    ("average_total_pressure", "p_th", "gap"),
    ("beta_toroidal", "beta_T", "gap"),
    ("beta_total", "beta", "gap"),
    # cfspopcon normalises beta_p to B_pol_out_mid = mu0 I_p/(2 pi a); selecting
    # "Poloidal beta (cfspopcon)" in reactor.yaml fixed the normalisation (was
    # +117%); the residual ~15% is the pressure convention above.
    ("beta_poloidal", "beta_p", "gap"),
    ("normalized_beta", "beta_N", "gap"),
    ("P_LH_thresh", "P_LH", "match"),
    # -- resistive / collisionality chain (fixed by the impurity-mix Z_eff) --
    ("z_effective", "Z_eff", "match"),
    ("effective_collisionality", "effective_collisionality", "match"),
    ("nu_star", "nu_star", "match"),
    ("spitzer_resistivity", "spitzer_resistivity", "match"),
    ("neoclassical_loop_resistivity", "neoclassical_loop_resistivity", "match"),
    ("current_relaxation_time", "current_relaxation_time", "match"),
    ("loop_voltage", "loop_voltage", "match"),
    ("inductive_plasma_current", "inductive_plasma_current", "match"),
    ("rho_star", "rho_star", "match"),
    # -- inductances / flux (Barr chain) --
    ("external_flux", "external_flux", "match"),
    ("external_inductance", "external_inductance", "match"),
    ("invmu_0_dLedR", "invmu_0_dLedR", "match"),
    ("vertical_magnetic_field", "vertical_magnetic_field", "match"),
    ("poloidal_field_flux", "poloidal_field_flux", "match"),
    # -- separatrix / SOL upstream (median matches; grid edges diverge) --
    # P_sep median dropped 3.4% -> 0.3% once the line radiation was activated
    # on the radas Lz tables (P_sep = P_loss - P_rad on the reference's own
    # radiators).
    ("power_crossing_separatrix", "P_sep", "match"),
    ("q_perp", "q_perp", "match"),
    ("PB_over_R", "PB_over_R", "match"),
    ("PBpRnSq", "PBpRnSq", "match"),
    ("lambda_q", "lambda_q", "match"),
    # Inherits P_LH's ~9% systematic; the P_sep improvement above brought the
    # median from ~11% to ~8.5%, within tolerance.
    ("ratio_of_P_SOL_to_P_LH", "ratio_of_P_SOL_to_P_LH", "match"),
    # P_aux is the scan's free DOF; its median agrees (~8%) but it blows up
    # relative to cfspopcon at the handful of near-ignition cells where P_aux -> 0.
    ("P_auxiliary_absorbed", "P_aux", "match"),
    ("P_auxiliary_launched", "P_aux_launched", "match"),
    # -- SOL two-point model (fixed-target-Te, activated 2026-07-19) --
    # The extended two-point model runs on input.yaml's SOL block (nesep ratio,
    # KotovReiter momentum loss, target Te held at 25 eV; see reactor.yaml):
    # medians land <0.3% grid-wide.  The maxima concentrate at the cold
    # low-density fringe, where the required cooling fraction saturates at 1
    # and the n_e = 1e19 column no longer certifies (1047 -> 1029 cells).
    ("separatrix_electron_density", "n_sep", "match"),
    ("separatrix_electron_temp", "T_sep", "match"),
    ("q_parallel", "q_parallel", "match"),
    ("target_q_parallel", "target_q_parallel", "match"),
    ("target_electron_density", "target_electron_density", "match"),
    ("target_electron_flux", "target_electron_flux", "match"),
    ("SOL_power_loss_fraction", "SOL_power_loss_fraction", "match"),
    # -- fusion chain: ~6% median (PRD cell +0.8%).  The fuel profile now
    # carries its OWN Angioni ion peaking (~1.40; the pointwise n_i ~ n_e tie
    # is a weak default the fixture excludes, with quasineutrality anchored at
    # the averages), so the fuel densities and shapes both match cfspopcon's;
    # the residual is reactivity-table-vs-Bosch-Hale plus the non-DT channels
    # cfspopcon does not compute, larger at the cold grid edge.
    ("P_fusion", "P_fus", "fusion"),
    ("P_neutron", "P_neutron", "fusion"),
    ("P_alpha", "P_fus_DT_alpha", "fusion"),
    # Neutron production rate over every neutron-producing channel fusdb models
    # (D-T + D-D(He3+n) + 2x T-T); ~6% median like P_neutron, but diverges more
    # at the hot corner (max ~14%) because T-T counts double in rate-space
    # (2 neutrons/reaction) yet only ~0.5x in power-space, so cfspopcon's
    # DT-only reference falls further behind here than it does for P_neutron.
    ("neutron_rate", "neutron_rate", "fusion"),
    # Neutron power flux to the wall = P_neutron / A_p -- fusdb's existing
    # q_wall (neutron wall loading), the same quantity cfspopcon reports here.
    # The ~3.5% median is a partial cancellation of two ~7-10% effects, NOT
    # clean agreement: fusdb's P_neutron runs ~6% high (extra D-D/T-T channels)
    # while fusdb's own plasma surface area A_p (60.97 m^2) runs ~10% above
    # cfspopcon's (55.54 m^2), and the excess power nearly offsets the excess
    # area.  P_neutron / 55.54 matches the reference to ~0.25%, so the residual
    # is a plasma-surface-area convention difference, not neutron physics.
    # (That A_p gap -- 9.8% while V_p agrees to 0.9% -- is also the likely
    # source of the P_LH ~9% systematic; see the SPARC comparison memo.)
    ("neutron_power_flux_to_walls", "q_wall", "match"),
    # Ohmic chain: fixed by the cfspopcon beta_p convention (was -20% via the
    # bootstrap -> inductive-current path); now within 10% grid-wide.
    ("P_ohmic", "P_ohmic", "match"),
    # -- peaking cluster (electron + ion Angioni, decoupled): <1% grid-wide --
    ("electron_density_peaking", "density_peaking", "match"),
    ("ion_density_peaking", "ion_density_peaking", "match"),
    ("peak_electron_density", "n0", "match"),
    ("peak_pressure", "p_peak", "match"),
    # Total-vs-fuel-ion semantics like average_ion_density above (fusdb's
    # n_i_peak counts all ions): a flat ~7% offset, within tolerance.
    ("peak_fuel_ion_density", "n_i_peak", "match"),
    ("fusion_triple_product", "n_i_tau_E_T_i", "match"),
    # Q on cfspopcon's own definition (P_fus / (P_aux_launched + P_ohmic),
    # verified against dataset P_external): ~9% median with the decoupled ion
    # peaking + cfspopcon W/tau_E conventions (was 59%).  The max blows up at
    # the handful of near-ignition cells where P_aux -> 0 amplifies any
    # residual stored-energy difference.
    ("Q", "Q_cfspopcon", "match"),
    # Radiation runs on cfspopcon's composition (hydrogenic bremsstrahlung +
    # synchrotron + line radiation from the input.yaml He/O/W concentrations)
    # with the radas 2-D Lz(Te, ne) tables evaluated pointwise -- the
    # reference's own radiation method: median ~3% (was ~53% with no line
    # radiation, ~14% on the Mavrin fits), 90th percentile ~4%.  The max
    # (~160%) is confined to the lowest-density grid column (n_e = 1e19),
    # where P_rad is ~0.1 MW, synchrotron-dominated, and the ~0.06 MW absolute
    # difference inflates the relative error.
    ("core_radiated_power_fraction", "f_rad", "match"),
    ("P_radiation", "P_rad", "match"),
    # -- known modelling gaps (reported, xfail at 10%) --
    # f_BS scales with beta_p, which carries the pressure-convention gap above.
    ("bootstrap_fraction", "f_BS", "gap"),
    ("flux_needed_from_CS_over_rampup", "flux_needed_from_CS_over_rampup", "gap"),
    # -- not produced by the scan fixture --
    # Lengyel chain stays off: L_int's canonical scale (~1e-28) is unsolvable
    # as a least-squares DOF, and fusdb's Mavrin-coronal L_int differs from
    # the reference's radas-noncoronal one anyway (see reactor.yaml).
    ("edge_impurity_concentration", "edge_impurity_concentration", "missing"),
    # Needs total_flux_available_from_CS (35 Wb in input.yaml); the flux chain
    # hung reconcile when last tried (2026-07) -- retest before activating.
    ("max_flattop_duration", "max_flattop_duration", "missing"),
    # Needs minimum_core_radiated_fraction (0.0 in input.yaml, so the
    # reference field is identically zero and uncomparable anyway).
    ("min_P_radiation", "min_P_radiation", "missing"),
]


def _attr_str(value) -> str:
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


def load_dataset():
    """Open the cfspopcon reference dataset (netCDF4/HDF5) with ``h5py``."""
    import h5py  # test-only dependency; imported lazily so collection never fails

    return h5py.File(DATASET_PATH, "r")


def dataset_grid(handle, dataset_name: str) -> np.ndarray | None:
    """Return one reference field as a ``(NY, NX)`` grid in fusdb units.

    cfspopcon stores grids as ``(n_e, T_e)`` = ``(40, 30)`` (some transposed);
    fusdb's popcon returns ``(T_e, n_e)`` = ``(30, 40)``, so everything is
    reduced to the fusdb ``(NY, NX)`` layout here.  1-D fields (varying along a
    single axis) are broadcast across the other.  Returns ``None`` if the field
    is absent or carries an unmapped unit.
    """
    if dataset_name not in handle:
        return None
    node = handle[dataset_name]
    unit = _attr_str(node.attrs.get("units", "dimensionless"))
    if unit not in UNIT_SCALE:
        return None
    arr = np.asarray(node[()], dtype=float) * UNIT_SCALE[unit]
    if arr.shape == (NX, NY):
        return arr.T
    if arr.shape == (NY, NX):
        return arr
    if arr.shape == (NX,):
        return np.broadcast_to(arr[None, :], (NY, NX)).copy()
    if arr.shape == (NY,):
        return np.broadcast_to(arr[:, None], (NY, NX)).copy()
    return None


def run_scan(outputs=None) -> dict:
    """Run a single-regime ``h_mode`` popcon scan for the faithful comparison.

    ``Reactor.popcon`` now selects the confinement regime per point automatically
    (H-mode where sustainable, L-mode below the L-H threshold); but cfspopcon's
    reference applies the single ITER98y2 (H-mode) scaling across the whole grid,
    so the value-by-value comparison against ``dataset.nc`` is taken against the
    forced-``h_mode`` scan (the reactor's own regime-cloning API).  Use
    :func:`run_scan_multiregime` for the native automatic-regime scan.
    """
    reactor = Reactor.from_yaml(REACTOR_YAML)
    clone = reactor._clone_for_regime("h_mode", include_guards=False)
    return clone._run_once("popcon", x=GRID_X, y=GRID_Y, outputs=outputs)


def run_scan_multiregime(outputs=None) -> dict:
    """Native automatic-per-point-regime popcon scan.

    Thin wrapper over ``Reactor.popcon``, which now assigns each grid point its
    own confinement regime (see ``Reactor._run_popcon_auto_regime``).  The result
    carries ``result["popcon"]["regime_index"]`` / ``["regime_names"]`` marking
    the regime each cell was solved in.  L-mode cells use the L-mode tau_E
    scaling, not cfspopcon's H-mode ITER98y2, so their values diverge from
    ``dataset.nc`` there by construction even though the domain now matches.
    """
    reactor = Reactor.from_yaml(REACTOR_YAML)
    return reactor.popcon(x=GRID_X, y=GRID_Y, outputs=outputs)


@dataclass
class FieldComparison:
    dataset_name: str
    fusdb_name: str
    category: str
    n_cells: int          # certified cells with a finite comparison on both sides
    median_rel: float     # median relative error over those cells (NaN if none)
    max_rel: float
    frac_within_10: float  # fraction of those cells within 10%

    @property
    def computed(self) -> bool:
        return self.n_cells > 0


def compare(result: dict, handle) -> list[FieldComparison]:
    """Compare a scan result against the reference for every field in ``FIELDS``."""
    payload = result["popcon"]
    ok = payload["success"]
    fields = payload["fields"]
    out: list[FieldComparison] = []
    for dataset_name, fusdb_name, category in FIELDS:
        canonical = VARIABLES.resolve(fusdb_name) if _resolvable(fusdb_name) else fusdb_name
        fusdb_grid = fields.get(canonical)
        ref = dataset_grid(handle, dataset_name)
        if fusdb_grid is None or ref is None:
            out.append(FieldComparison(dataset_name, fusdb_name, category, 0, np.nan, np.nan, 0.0))
            continue
        # Exclude cells where the reference is ~zero *relative to the field's own
        # scale* (a genuine divide-by-zero in the relative error), not a fixed SI
        # floor -- some fields are uniformly tiny in SI (e.g. PBpRnSq ~1e-34).
        finite_ref = ref[np.isfinite(ref)]
        scale = np.max(np.abs(finite_ref)) if finite_ref.size else 0.0
        floor = 1e-9 * scale
        mask = ok & np.isfinite(fusdb_grid) & np.isfinite(ref) & (np.abs(ref) > floor)
        if not mask.any():
            out.append(FieldComparison(dataset_name, fusdb_name, category, 0, np.nan, np.nan, 0.0))
            continue
        rel = np.abs(fusdb_grid[mask] - ref[mask]) / np.abs(ref[mask])
        out.append(
            FieldComparison(
                dataset_name, fusdb_name, category, int(mask.sum()),
                float(np.median(rel)), float(np.max(rel)), float(np.mean(rel < 0.10)),
            )
        )
    return out


def _resolvable(name: str) -> bool:
    try:
        VARIABLES.resolve(name)
        return True
    except Exception:
        return False


# Output list the scan needs to compute every mapped field (skips the ``missing``
# targets that would otherwise emit "not derivable" warnings).
SCAN_OUTPUTS = tuple(
    sorted({VARIABLES.resolve(f) for _d, f, cat in FIELDS if cat != "missing" and _resolvable(f)})
)


# ── T-n contour overlay ──────────────────────────────────────────────────────
#
# Quantities drawn on the operating-space overlay: (fusdb variable, dataset
# field, axis label, display factor onto the label's unit, contour levels).
# fusdb popcon fields and dataset_grid both come out in fusdb canonical units
# (W, s, dimensionless), so one display factor serves both sides.  Q compares
# ``Q_cfspopcon`` (cfspopcon's launched-power definition, P_fus /
# (P_aux_launched + P_ohmic)) so both sides share one definition.
OVERLAY_QUANTITIES = [
    ("P_fus", "P_fusion", "$P_{fus}$ [MW]", 1.0e-6, [25.0, 50.0, 100.0, 150.0]),
    ("Q_cfspopcon", "Q", "$Q$ (launched)", 1.0, [1.0, 2.0, 5.0]),
    ("P_aux", "P_auxiliary_absorbed", "$P_{aux}$ [MW]", 1.0e-6, [5.0, 15.0, 25.0]),
    ("ratio_of_P_SOL_to_P_LH", "ratio_of_P_SOL_to_P_LH", "$P_{SOL}/P_{LH}$", 1.0, [1.0]),
    ("max_flattop_duration", "max_flattop_duration", "$t_{flattop}$ [s]", 1.0, [3.0, 10.0, 30.0]),
]


def contour_overlay(result: dict, handle, ax=None):
    """Overlay fusdb (solid) and cfspopcon (dashed) contours on the T-n plane.

    Draws the operating-space contours for fusion power, Q, auxiliary power, the
    L-H ratio and the flat-top duration.  Each quantity gets one colour; fusdb is
    solid, cfspopcon dashed.

    Two features make the comparison honest rather than misleading:

    * The region fusdb **does not certify** (mostly ``T_e < 9`` keV, where the
      H-mode power/confinement balance has no valid operating point) is shaded;
      fusdb contours necessarily stop at its edge while cfspopcon's continue, so
      the shading marks *why* the solid lines are truncated rather than leaving
      an unexplained gap.
    * ``Q`` is compared as ``Q_cfspopcon`` -- cfspopcon's own launched-power
      definition, P_fus / (P_aux_launched + P_ohmic) -- not fusdb's
      absorbed-power ``Q_sci``, which diverges to infinity where P_aux -> 0.

    Quantities fusdb does not compute on the scan (e.g. ``t_flattop``) show only
    the cfspopcon (dashed) contour -- the mismatch the notebook is meant to expose.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    payload = result["popcon"]
    x = payload["x"]["values"] / 1.0e20   # 1e20 m^-3
    y = payload["y"]["values"]            # keV
    ok = payload["success"]
    fields = payload["fields"]
    regime_index = payload.get("regime_index")
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8.5, 6.5))

    style_extra = []
    if regime_index is not None:
        # Per-point regime fill: colour the background by the regime each cell
        # was solved in (index into regime_names; -1 = no regime certified).
        names = payload.get("regime_names", ())
        palette = {"h_mode": "#dbe9ff", "l_mode": "#ffe7cc", "ohmic_mode": "#dcf5dc", "i_mode": "#f0dcf5"}
        used = [i for i in range(len(names)) if np.any(regime_index == i)]
        cmap = ListedColormap([palette.get(names[i], "0.8") for i in used])
        norm = BoundaryNorm([*range(len(used) + 1)], cmap.N)
        shown = np.full(regime_index.shape, np.nan)
        for new, orig in enumerate(used):
            shown[regime_index == orig] = new
        ax.pcolormesh(x, y, np.ma.masked_invalid(shown), cmap=cmap, norm=norm, shading="nearest", zorder=0, alpha=0.65)
        style_extra = [Patch(facecolor=palette.get(names[i], "0.8"), label=names[i].replace("_", "-")) for i in used]
        if np.any(regime_index < 0):
            style_extra.append(Patch(facecolor="white", edgecolor="0.6", label="no regime"))
    else:
        # Shade the region fusdb cannot certify (contourf of the boolean mask).
        ax.contourf(x, y, (~ok).astype(float), levels=[0.5, 1.5], colors=["0.85"], alpha=0.7, zorder=0)
        style_extra = [Patch(facecolor="0.85", alpha=0.7, label="fusdb uncertified")]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_handles = []
    for i, (fusdb_name, dataset_name, label, scale, levels) in enumerate(OVERLAY_QUANTITIES):
        color = colors[i % len(colors)]
        canonical = VARIABLES.resolve(fusdb_name) if _resolvable(fusdb_name) else fusdb_name
        fusdb_grid = fields.get(canonical)
        # P_SOL/P_LH is the L-H *decision* variable, and P_sep is regime-dependent
        # (H-mode tau_E >> L-mode -> P_sep_h << P_sep_l at the same point).  On the
        # auto-regime overlay the composite field mixes the two solves, so its
        # ``= 1`` contour lands inside the L-fallback (bistable) band, not on the
        # regime boundary.  Plot it instead from the H-mode reference solve (the
        # variable the boundary is actually defined by), so it stays coincident
        # with the H/L background and matches cfspopcon's single-solve convention.
        if regime_index is not None and fusdb_name == "ratio_of_P_SOL_to_P_LH":
            reference = payload.get("lh_ratio_reference")
            if reference is not None:
                fusdb_grid = reference
        drew = False
        if fusdb_grid is not None and np.isfinite(fusdb_grid[ok]).any():
            masked = np.where(ok, fusdb_grid, np.nan) * scale
            cs = ax.contour(x, y, masked, levels=levels, colors=color, linestyles="solid", linewidths=1.7, zorder=3)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
            drew = True
        ref = dataset_grid(handle, dataset_name)
        if ref is not None:
            cs = ax.contour(x, y, ref * scale, levels=levels, colors=color, linestyles="dashed", linewidths=1.4, zorder=2)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
            drew = True
        if drew:
            legend_handles.append(Line2D([0], [0], color=color, lw=2, label=label))

    # Style legend: colour = quantity, line style = code, plus the shaded region.
    style_handles = [
        Line2D([0], [0], color="0.3", lw=2, ls="solid", label="fusdb"),
        Line2D([0], [0], color="0.3", lw=2, ls="dashed", label="cfspopcon"),
        *style_extra,
    ]
    first = ax.legend(handles=legend_handles, loc="upper left", fontsize=8, title="quantity")
    ax.add_artist(first)
    ax.legend(handles=style_handles, loc="lower right", fontsize=8, title="source")
    ax.set_xlabel(r"$\langle n_e \rangle$ [$10^{20}\,\mathrm{m^{-3}}$]")
    ax.set_ylabel(r"$\langle T_e \rangle$ [keV]")
    ax.set_title("SPARC PRD operating space: fusdb (solid) vs cfspopcon (dashed)")
    return ax
