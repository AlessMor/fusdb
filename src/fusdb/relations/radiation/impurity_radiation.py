"""Impurity line-radiation relations (Mavrin + Post-Jensen cooling curves).

Adapted from cfspopcon; see README.md section "Third-party Notices". The
polynomial-fit methods need no ADAS/radas data, so the coefficients live in
``registry/atomic_radiation/`` and are read by the loader below. Three methods
output ``P_line`` (gated, Mavrin coronal default): Mavrin coronal (2018),
Mavrin non-coronal (2017, ne*tau dependent), and Post-Jensen (1977).

Each ``c_X`` is the impurity concentration n_X/n_e (default 0, absent species do
not contribute). Radiated power density is q = n_e^2 * sum_s(c_s * Lz_s),
integrated over volume (rho-uniform, consistent with the Bremsstrahlung/synchrotron
relations).

Unit/correctness notes (all # CHECK):
* Mavrin coronal: cfspopcon's body swaps the temp/density variable names; this uses
  the physically-correct Lz = fit(T_e[keV]).
* Mavrin non-coronal: cfspopcon labels the temperature bins "eV" but feeds T in keV
  (input_units=keV); replicated as-is (T in keV vs the bin borders). cfspopcon's
  qRad also omits the 1e38 (n19->m^-3) density factor that the coronal path applies;
  the physically-consistent n_e^2 [m^-3] factor is used here. Unverifiable without a
  cfspopcon install -- review.
* Post-Jensen: Lz fit is in erg*cm^3 (factor 1e-13 -> W*m^3); T in keV.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.polynomial.polynomial import polyval
from fusdb.utils import trapezoid

from fusdb.relation import relation

_DATA = Path(__file__).resolve().parents[2] / "registry" / "atomic_radiation"
_MAVRIN_T_MIN, _MAVRIN_T_MAX = 0.1, 100.0  # Mavrin 2018 coronal validity [keV]
_ERG_CM3_TO_W_M3 = 1.0e-13  # Post-Jensen Lz units


@lru_cache(maxsize=4)
def _load_raw(filename: str) -> dict:
    """Load a cooling-curve coefficient table from registry/atomic_radiation/."""
    return yaml.safe_load((_DATA / filename).read_text())


def _binned_log10_Lz(Te: np.ndarray, bins: Any, radc: Any) -> np.ndarray:
    """Evaluate log10(Lz) over piecewise 1-D polynomial temperature bins."""
    Tlog = np.log10(Te)
    out = np.zeros_like(Te)
    for i, row in enumerate(radc):
        mask = (Te >= bins[i]) & (Te < bins[i + 1])
        if np.any(mask):
            out[mask] = polyval(Tlog[mask], row)
    return out


@relation(
    name="Impurity line radiation (Mavrin coronal)",
    tags=("power_balance",),
    outputs="P_line",
)
def calc_impurity_line_radiation_mavrin_coronal(
    n_e: Any, T_e: Any, rho: Any, V_p: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
) -> Any:
    """Total impurity line-radiated power, Mavrin 2018 coronal cooling rates.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    coeffs = _load_raw("mavrin_coronal.yaml")
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    Te = np.clip(np.asarray(T_e, dtype=float), _MAVRIN_T_MIN, _MAVRIN_T_MAX)  # keV
    c_times_Lz = np.zeros_like(Te)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        entry = coeffs[symbol]
        Lz = 10.0 ** _binned_log10_Lz(Te, entry["temperature_bin_borders"], entry["radc"])  # [W*m^3]
        c_times_Lz = c_times_Lz + concentration * Lz
    q_rad = np.nan_to_num(np.asarray(n_e, dtype=float) ** 2 * c_times_Lz, nan=0.0)
    return V_p * trapezoid(q_rad, x=rho)


@relation(
    name="Impurity line radiation (Post-Jensen)",
    tags=("power_balance",),
    outputs="P_line",
)
def calc_impurity_line_radiation_post_jensen(
    n_e: Any, T_e: Any, rho: Any, V_p: Any,
    c_He: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_O: Any = 0.0, c_Ar: Any = 0.0, c_W: Any = 0.0,
) -> Any:
    """Total impurity line-radiated power, Post & Jensen 1977 cooling rates.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    Supports the species shared with the Mavrin set (He, Be, C, O, Ar, W).
    """
    # CHECK
    data = _load_raw("post_jensen.yaml")
    bins = data["_temperature_bin_borders"]
    concentrations = {"He": c_He, "Be": c_Be, "C": c_C, "O": c_O, "Ar": c_Ar, "W": c_W}
    n_e_arr = np.asarray(n_e, dtype=float)
    c_times_Lz = np.zeros_like(n_e_arr, dtype=float)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        entry = data[symbol]
        Te = np.clip(np.asarray(T_e, dtype=float), entry["Tmin"], 100.0)  # keV
        Lz = 10.0 ** _binned_log10_Lz(Te, bins, entry["radc"]) * _ERG_CM3_TO_W_M3  # [W*m^3]
        c_times_Lz = c_times_Lz + concentration * Lz
    q_rad = np.nan_to_num(n_e_arr ** 2 * c_times_Lz, nan=0.0)
    return V_p * trapezoid(q_rad, x=rho)


@relation(
    name="Impurity line radiation (Mavrin noncoronal)",
    tags=("power_balance",),
    outputs="P_line",
)
def calc_impurity_line_radiation_mavrin_noncoronal(
    n_e: Any, T_e: Any, rho: Any, V_p: Any, impurity_residence_time: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0,
    c_N: Any = 0.0, c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0,
) -> Any:
    """Total impurity line-radiated power, Mavrin 2017 non-coronal (ne*tau) cooling rates.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    2-D bicubic fit in (log10 T_e, log10(ne*tau/1e19)). See module-level # CHECK
    notes on the keV/eV bin labelling and density-factor discrepancies in cfspopcon.
    """
    # CHECK
    data = _load_raw("mavrin_noncoronal.yaml")
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C,
                      "N": c_N, "O": c_O, "Ne": c_Ne, "Ar": c_Ar}
    n_e_arr = np.asarray(n_e, dtype=float)
    ne_tau = n_e_arr * float(impurity_residence_time)
    with np.errstate(divide="ignore"):
        Y = np.minimum(np.log10(np.maximum(ne_tau, np.finfo(float).tiny) / 1.0e19), 0.0)
    c_times_Lz = np.zeros_like(n_e_arr, dtype=float)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        entry = data[symbol]
        bins = entry["temperature_bin_borders"]
        radc = entry["radc"]  # 10 rows (coeffs) x nbins
        Te = np.clip(np.asarray(T_e, dtype=float), bins[0], bins[-1])
        X = np.log10(Te)
        log10_Lz = np.zeros_like(Te)
        for j in range(len(bins) - 1):
            mask = (Te >= bins[j]) & (Te <= bins[j + 1])
            if np.any(mask):
                rc = [radc[k][j] for k in range(10)]
                Xm, Ym = X[mask], Y[mask]
                log10_Lz[mask] = (
                    rc[0] + rc[1] * Xm + rc[2] * Ym + rc[3] * Xm**2 + rc[4] * Xm * Ym
                    + rc[5] * Ym**2 + rc[6] * Xm**3 + rc[7] * Xm**2 * Ym + rc[8] * Xm * Ym**2 + rc[9] * Ym**3
                )
        c_times_Lz = c_times_Lz + concentration * 10.0**log10_Lz  # [W*m^3]
    q_rad = np.nan_to_num(n_e_arr ** 2 * c_times_Lz, nan=0.0)
    return V_p * trapezoid(q_rad, x=rho)


# ── PROCESS tabulated coronal Lz (impurity_radiation.py pimpden) ──────────────
# Adapted from PROCESS; see README.md section "Third-party Notices".
# PROCESS radiates impurities from tabulated coronal-equilibrium Lz(Te) curves
# (ADAS acd85/scd85/plt89/prb89, the "infinite confinement" column of
# data/lz_non_corona_14_elements/), log-log interpolated in temperature -- an
# alternative to the Mavrin/Post-Jensen polynomial fits. The tables were
# extracted programmatically (no transcription) into
# registry/atomic_radiation/process_coronal_lz.yaml. Gated (Mavrin coronal
# stays the default). Species: the 10 for which PROCESS has data AND fusdb has a
# concentration variable (no Li in PROCESS's set).

# PROCESS overwrites pimpden with the *raw* endpoint Lz outside the table range
# (unit-inconsistent -- it drops the n_e^2 factor -- but negligible there since
# Bremsstrahlung dominates at high Te). This port instead relies on np.interp's
# natural endpoint clamping so the n_e^2 scaling is preserved. (# CHECK)
_PROCESS_LZ_SPECIES = ("He", "Be", "C", "N", "O", "Ne", "Ar", "Kr", "Xe", "W")


def _process_coronal_Lz(symbol: str, Te_keV: np.ndarray) -> np.ndarray:
    """PROCESS coronal Lz [W*m^3] for one species: log-log interpolation of the
    tabulated Lz(Te) curve (PROCESS ``pimpden``)."""
    entry = _load_raw("process_coronal_lz.yaml")[symbol]
    T_tab = np.asarray(entry["temperature_keV"], dtype=float)
    Lz_tab = np.asarray(entry["Lz_Wm3"], dtype=float)
    # Clip to the table range: this both implements the endpoint clamp (Lz[0]
    # below / Lz[-1] above, n_e^2-scaled) and avoids log(0) where fusdb's
    # parabolic T_e reaches ~0 at the edge.
    Te = np.clip(np.asarray(Te_keV, dtype=float), T_tab[0], T_tab[-1])
    return np.exp(np.interp(np.log(Te), np.log(T_tab), np.log(Lz_tab)))


@relation(
    name="Impurity line radiation (PROCESS coronal tables)",
    tags=("power_balance", "process"),
    outputs="P_line",
)
def calc_impurity_line_radiation_process_coronal(
    n_e: Any, T_e: Any, rho: Any, V_p: Any,
    c_He: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0, c_O: Any = 0.0,
    c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
) -> Any:
    """Total impurity line-radiated power from PROCESS's tabulated coronal Lz
    cooling curves.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    Same rho-integrated assembly as the Mavrin relations (q = n_e^2 * sum_s
    c_s Lz_s, integrated V_p * trapezoid over rho) so it is a drop-in
    ``P_line`` alternative; only the Lz source differs.
    """
    # CHECK
    concentrations = {"He": c_He, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    Te = np.asarray(T_e, dtype=float)
    c_times_Lz = np.zeros_like(Te)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        Lz = _process_coronal_Lz(symbol, Te)  # [W*m^3]
        c_times_Lz = c_times_Lz + concentration * Lz
    q_rad = np.nan_to_num(np.asarray(n_e, dtype=float) ** 2 * c_times_Lz, nan=0.0)
    return V_p * trapezoid(q_rad, x=rho)


# ── Edge impurity seeding (Lengyel model, cfspopcon impurities/edge_radiator_conc) ──
# ADAS-free: the cooling-curve integral L_int = int(Lz*sqrt(Te) dTe) is evaluated
# numerically over the Mavrin coronal Lz curve (cfspopcon uses the radas noncoronal
# Lz). T enters in eV (the Lengyel formula and kappa_e0 are eV-based). Unverified
# against cfspopcon -- review (# CHECK).

_KEV_TO_EV = 1.0e3


def _mavrin_coronal_Lz(symbol: str, Te_keV: np.ndarray) -> np.ndarray:
    """Mavrin 2018 coronal Lz [W*m^3] for one species over a keV temperature grid."""
    entry = _load_raw("mavrin_coronal.yaml")[symbol]
    Te = np.clip(Te_keV, _MAVRIN_T_MIN, _MAVRIN_T_MAX)
    return 10.0 ** _binned_log10_Lz(Te, entry["temperature_bin_borders"], entry["radc"])


def _cooling_integral(symbol: str, target_electron_temp: Any, T_sep: Any) -> Any:
    """L_int = int_{target}^{sep} Lz(Te)*sqrt(Te) dTe [W*m^3*eV^1.5], Te in eV."""
    lo = float(target_electron_temp) * _KEV_TO_EV
    hi = float(T_sep) * _KEV_TO_EV
    if hi <= lo:
        return 0.0
    grid_eV = np.linspace(lo, hi, 65)
    Lz = _mavrin_coronal_Lz(symbol, grid_eV / _KEV_TO_EV)
    return float(trapezoid(Lz * np.sqrt(grid_eV), x=grid_eV))


for _sym in ("N", "Ne", "Ar"):
    def _make_L_int(symbol: str) -> Any:
        def _func(target_electron_temp: Any, T_sep: Any) -> Any:
            # CHECK
            return _cooling_integral(symbol, target_electron_temp, T_sep)
        _func.__name__ = f"calc_L_int_cooling_integral_{symbol}"
        _func.__doc__ = (
            f"Cooling-curve integral int(Lz*sqrt(Te) dTe) for {symbol} (Mavrin coronal Lz).\n\n"
            'Adapted from cfspopcon; see README.md section "Third-party Notices".'
        )
        return _func

    relation(
        name=f"L_int cooling integral {_sym}",
        tags=("power_exhaust", "tokamak"),
        outputs="L_int",
    )(_make_L_int(_sym))


@relation(
    name="Edge impurity concentration (Lengyel)",
    tags=("power_exhaust", "tokamak"),
    outputs="edge_impurity_concentration",
)
def calc_edge_impurity_concentration(
    q_parallel: Any,
    SOL_power_loss_fraction: Any,
    n_sep: Any,
    T_sep: Any,
    kappa_e0: Any,
    L_int: Any,
    lengyel_overestimation_factor: Any = 1.0,
) -> Any:
    """Edge impurity concentration to cool the SOL (Lengyel model).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    Lengyel 1981 / Moulton 2021. ``kappa_e0`` and ``L_int`` are eV-based, so the
    separatrix temperature is converted keV->eV.
    """
    # CHECK
    T_sep_eV = T_sep * _KEV_TO_EV
    numerator = q_parallel**2 - ((1.0 - SOL_power_loss_fraction) * q_parallel) ** 2
    denominator = 2.0 * kappa_e0 * (n_sep * T_sep_eV) ** 2 * L_int
    return numerator / denominator / lengyel_overestimation_factor


@relation(
    name="Edge impurity concentration in core",
    tags=("power_exhaust", "tokamak"),
    outputs="edge_impurity_concentration_in_core",
)
def calc_edge_impurity_concentration_in_core(edge_impurity_concentration: Any, edge_impurity_enrichment: Any) -> Any:
    """Edge impurity concentration referred to the core (divided by the enrichment).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return edge_impurity_concentration / edge_impurity_enrichment
