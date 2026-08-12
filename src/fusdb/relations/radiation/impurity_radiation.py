"""Impurity line-radiation relations (Mavrin + Post-Jensen cooling curves).

Adapted from cfspopcon; see README.md section "Third-party Notices". The
polynomial-fit methods need no ADAS/radas data, so their per-species datasets
are resolved through ``fusdb.registry.dataset``. Five methods output the
impurity cooling power: Mavrin coronal/non-coronal, Post-Jensen, PROCESS
coronal tables, and radas (OpenADAS) coronal tables.

Each ``c_X`` is the impurity concentration n_X/n_e. Local radiated-power
densities are integrated over physical volume using ``w_V`` when supplied;
omitting it retains fusdb's historical self-similar volume measure.
"""

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.polynomial.polynomial import polyval
from fusdb.utils import trapezoid, volume_average

from fusdb.relation import relation
from fusdb.registry import SPECIES
from fusdb.registry.dataset import load_dataset

_MAVRIN_T_MIN, _MAVRIN_T_MAX = 0.1, 100.0
_ERG_CM3_TO_W_M3 = 1.0e-13


@lru_cache(maxsize=None)
def _load_radiation_dataset(datatype: str, source: str, symbol: str) -> dict[str, Any]:
    """Load one per-species radiation dataset through the dataset registry."""
    dataset_id = f"{datatype}_{source}_{symbol}"
    return load_dataset(dataset_id, expected_datatype=datatype).data


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
    outputs="P_cool_imp",
)
def calc_impurity_line_radiation_mavrin_coronal(
    n_e: Any, T_e: Any, rho: Any, V_p: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0,
    c_W: Any = 0.0, w_V: Any = None,
) -> Any:
    """Total impurity radiated power, Mavrin 2018 coronal cooling rates."""
    # CHECK
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    Te = np.clip(np.asarray(T_e, dtype=float), _MAVRIN_T_MIN, _MAVRIN_T_MAX)
    c_times_Lz = np.zeros_like(Te)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        entry = _load_radiation_dataset("polynomialfit", "mavrin_coronal", symbol)
        Lz = 10.0 ** _binned_log10_Lz(Te, entry["temperature_bin_borders"], entry["radc"])
        c_times_Lz = c_times_Lz + concentration * Lz
    q_rad = np.nan_to_num(np.asarray(n_e, dtype=float) ** 2 * c_times_Lz, nan=0.0)
    return V_p * volume_average(q_rad, rho, weight=w_V)


@relation(
    name="Impurity line radiation (Post-Jensen)",
    tags=("power_balance",),
    outputs="P_cool_imp",
)
def calc_impurity_line_radiation_post_jensen(
    n_e: Any, T_e: Any, rho: Any, V_p: Any,
    c_He: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_O: Any = 0.0, c_Ar: Any = 0.0,
    c_W: Any = 0.0, w_V: Any = None,
) -> Any:
    """Total impurity radiated power, Post & Jensen 1977 cooling rates."""
    # CHECK
    concentrations = {"He": c_He, "Be": c_Be, "C": c_C, "O": c_O, "Ar": c_Ar, "W": c_W}
    n_e_arr = np.asarray(n_e, dtype=float)
    c_times_Lz = np.zeros_like(n_e_arr, dtype=float)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        entry = _load_radiation_dataset("polynomialfit", "post_jensen", symbol)
        bins = entry["temperature_bin_borders"]
        Te = np.clip(np.asarray(T_e, dtype=float), entry["Tmin"], 100.0)
        Lz = 10.0 ** _binned_log10_Lz(Te, bins, entry["radc"]) * _ERG_CM3_TO_W_M3
        c_times_Lz = c_times_Lz + concentration * Lz
    q_rad = np.nan_to_num(n_e_arr ** 2 * c_times_Lz, nan=0.0)
    return V_p * volume_average(q_rad, rho, weight=w_V)


@relation(
    name="Impurity line radiation (Mavrin noncoronal)",
    tags=("power_balance",),
    outputs="P_cool_imp",
)
def calc_impurity_line_radiation_mavrin_noncoronal(
    n_e: Any, T_e: Any, rho: Any, V_p: Any, impurity_residence_time: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0,
    c_N: Any = 0.0, c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0,
    w_V: Any = None,
) -> Any:
    """Total impurity radiated power, Mavrin 2017 non-coronal cooling rates."""
    # CHECK
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
        entry = _load_radiation_dataset("polynomialfit", "mavrin_noncoronal", symbol)
        bins = entry["temperature_bin_borders"]
        radc = entry["radc"]
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
                    + rc[5] * Ym**2 + rc[6] * Xm**3 + rc[7] * Xm**2 * Ym
                    + rc[8] * Xm * Ym**2 + rc[9] * Ym**3
                )
        c_times_Lz = c_times_Lz + concentration * 10.0**log10_Lz
    q_rad = np.nan_to_num(n_e_arr ** 2 * c_times_Lz, nan=0.0)
    return V_p * volume_average(q_rad, rho, weight=w_V)


def _process_coronal_Lz(symbol: str, Te_keV: np.ndarray) -> np.ndarray:
    """PROCESS coronal Lz [W*m^3] for one species."""
    entry = _load_radiation_dataset("coolingcurve", "PROCESS_coronal", symbol)
    T_tab = np.asarray(entry["temperature_keV"], dtype=float)
    Lz_tab = np.asarray(entry["Lz_Wm3"], dtype=float)
    Te = np.clip(np.asarray(Te_keV, dtype=float), T_tab[0], T_tab[-1])
    return np.exp(np.interp(np.log(Te), np.log(T_tab), np.log(Lz_tab)))


@relation(
    name="Impurity line radiation (PROCESS coronal tables)",
    tags=("power_balance", "process"),
    outputs="P_cool_imp",
)
def calc_impurity_line_radiation_process_coronal(
    n_e: Any, T_e: Any, rho: Any, V_p: Any,
    c_He: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0, c_O: Any = 0.0,
    c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0,
    c_W: Any = 0.0, w_V: Any = None,
) -> Any:
    """Total impurity radiated power from PROCESS's tabulated coronal Lz curves."""
    # CHECK
    concentrations = {"He": c_He, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    Te = np.asarray(T_e, dtype=float)
    c_times_Lz = np.zeros_like(Te)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        Lz = _process_coronal_Lz(symbol, Te)
        c_times_Lz = c_times_Lz + concentration * Lz
    q_rad = np.nan_to_num(np.asarray(n_e, dtype=float) ** 2 * c_times_Lz, nan=0.0)
    return V_p * volume_average(q_rad, rho, weight=w_V)


@relation(
    name="Species-sum radiated power (PROCESS coronal tables)",
    tags=("power_balance", "process"),
    outputs="P_rad_species",
)
def calc_radiated_power_species_sum_process_coronal(
    n_e: Any, T_e: Any, rho: Any, V_p: Any,
    c_H: Any = 0.0, c_He: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0,
    c_W: Any = 0.0, w_V: Any = None,
) -> Any:
    """Total PROCESS-table radiated power over fuel, ash, and impurities."""
    # CHECK
    concentrations = {"H": c_H, "He": c_He, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    Te = np.asarray(T_e, dtype=float)
    c_times_Lz = np.zeros_like(Te)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        Lz = _process_coronal_Lz(symbol, Te)
        c_times_Lz = c_times_Lz + concentration * Lz
    q_rad = np.nan_to_num(np.asarray(n_e, dtype=float) ** 2 * c_times_Lz, nan=0.0)
    return V_p * volume_average(q_rad, rho, weight=w_V)


_RADAS_LZ_SPECIES = SPECIES.with_atomic_data("coolingcurve_radas_coronal")


@lru_cache(maxsize=None)
def _radas_coronal_spline(symbol: str) -> Any:
    """Log-log bicubic spline over the 2-D Lz table, plus coordinate bounds."""
    from scipy.interpolate import RectBivariateSpline

    entry = _load_radiation_dataset("coolingcurve", "radas_coronal", symbol)
    logT = np.log10(np.asarray(entry["temperature_keV"], dtype=float))
    logn = np.log10(np.asarray(entry["electron_density_m3"], dtype=float))
    logLz = np.log10(np.asarray(entry["Lz_Wm3"], dtype=float))
    spline = RectBivariateSpline(logT, logn, logLz)
    return spline, (logT[0], logT[-1]), (logn[0], logn[-1])


_RADAS_MEANCHARGE_SPECIES = SPECIES.with_atomic_data("meancharge_radas_coronal")


@lru_cache(maxsize=None)
def _radas_meancharge_spline(symbol: str) -> Any:
    """Bicubic spline over the 2-D mean-charge table, plus coordinate bounds."""
    from scipy.interpolate import RectBivariateSpline

    entry = _load_radiation_dataset("meancharge", "radas_coronal", symbol)
    logT = np.log10(np.asarray(entry["temperature_keV"], dtype=float))
    logn = np.log10(np.asarray(entry["electron_density_m3"], dtype=float))
    zbar = np.asarray(entry["mean_charge"], dtype=float)
    spline = RectBivariateSpline(logT, logn, zbar)
    return spline, (logT[0], logT[-1]), (logn[0], logn[-1]), float(entry["atomic_number"])


def radas_mean_charge(symbol: str, Te_keV: Any, n_e: Any) -> np.ndarray:
    """radas coronal mean charge Zbar(T_e, n_e) for one species."""
    spline, (logT_lo, logT_hi), (logn_lo, logn_hi), Z = _radas_meancharge_spline(symbol)
    with np.errstate(divide="ignore"):
        lT = np.log10(np.asarray(Te_keV, dtype=float))
        ln = np.log10(np.asarray(n_e, dtype=float))
    lT, ln = np.broadcast_arrays(lT, ln)
    lT = np.clip(lT, logT_lo, logT_hi)
    ln = np.clip(ln, logn_lo, logn_hi)
    val = spline(np.ravel(lT), np.ravel(ln), grid=False).reshape(np.shape(lT))
    return np.clip(val, 0.0, Z)


_RADAS_MEANSQ_SPECIES = SPECIES.with_atomic_data("meansquarecharge_radas_coronal")


@lru_cache(maxsize=None)
def _radas_meansq_spline(symbol: str) -> Any:
    """Bicubic spline over the 2-D <q^2> table, plus coordinate bounds."""
    from scipy.interpolate import RectBivariateSpline

    entry = _load_radiation_dataset("meansquarecharge", "radas_coronal", symbol)
    logT = np.log10(np.asarray(entry["temperature_keV"], dtype=float))
    logn = np.log10(np.asarray(entry["electron_density_m3"], dtype=float))
    q2 = np.asarray(entry["mean_square_charge"], dtype=float)
    spline = RectBivariateSpline(logT, logn, q2)
    zsq_max = float(entry["atomic_number"]) ** 2
    return spline, (logT[0], logT[-1]), (logn[0], logn[-1]), zsq_max


def radas_mean_square_charge(symbol: str, Te_keV: Any, n_e: Any) -> np.ndarray:
    """radas coronal mean-square charge <q^2>(T_e, n_e) for one species."""
    spline, (logT_lo, logT_hi), (logn_lo, logn_hi), zsq_max = _radas_meansq_spline(symbol)
    with np.errstate(divide="ignore"):
        lT = np.log10(np.asarray(Te_keV, dtype=float))
        ln = np.log10(np.asarray(n_e, dtype=float))
    lT, ln = np.broadcast_arrays(lT, ln)
    lT = np.clip(lT, logT_lo, logT_hi)
    ln = np.clip(ln, logn_lo, logn_hi)
    val = spline(np.ravel(lT), np.ravel(ln), grid=False).reshape(np.shape(lT))
    return np.clip(val, 0.0, zsq_max)


def _radas_coronal_Lz(symbol: str, Te_keV: Any, n_e: Any) -> np.ndarray:
    """radas coronal Lz [W*m^3] for one species at local T_e and n_e."""
    spline, (logT_lo, logT_hi), (logn_lo, logn_hi) = _radas_coronal_spline(symbol)
    with np.errstate(divide="ignore"):
        lT = np.log10(np.asarray(Te_keV, dtype=float))
        ln = np.log10(np.asarray(n_e, dtype=float))
    lT, ln = np.broadcast_arrays(lT, ln)
    lT = np.clip(lT, logT_lo, logT_hi)
    ln = np.clip(ln, logn_lo, logn_hi)
    log_val = spline(np.ravel(lT), np.ravel(ln), grid=False).reshape(np.shape(lT))
    return 10.0 ** log_val


def _make_Lz_relation(symbol: str) -> Any:
    def _func(T_e: Any, n_e: Any) -> Any:
        # CHECK
        return _radas_coronal_Lz(symbol, T_e, n_e)

    _func.__name__ = f"calc_Lz_radas_coronal_{symbol}"
    _func.__doc__ = (
        f"Coronal cooling factor Lz(T_e, n_e) for {symbol} from the radas (OpenADAS) tables.\n\n"
        'Adapted from cfspopcon; see README.md section "Third-party Notices".'
    )
    return _func


for _sym in _RADAS_LZ_SPECIES:
    relation(
        name=f"Radas coronal cooling factor {_sym}",
        tags=("power_balance",),
        outputs="Lz",
    )(_make_Lz_relation(_sym))


@relation(
    name="Impurity line radiation (radas coronal)",
    tags=("power_balance",),
    outputs="P_cool_imp",
)
def calc_impurity_line_radiation_radas_coronal(
    n_e: Any, T_e: Any, rho: Any, V_p: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Xe: Any = 0.0,
    c_W: Any = 0.0, w_V: Any = None,
) -> Any:
    """Total impurity radiated power from radas (OpenADAS) coronal Lz tables."""
    # CHECK
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N,
                      "O": c_O, "Ne": c_Ne, "Ar": c_Ar, "Xe": c_Xe, "W": c_W}
    Te = np.asarray(T_e, dtype=float)
    ne = np.asarray(n_e, dtype=float)
    c_times_Lz = np.zeros_like(Te)
    for symbol, concentration in concentrations.items():
        if np.all(np.asarray(concentration, dtype=float) == 0.0):
            continue
        Lz = _radas_coronal_Lz(symbol, Te, ne)
        c_times_Lz = c_times_Lz + concentration * Lz
    q_rad = np.nan_to_num(ne ** 2 * c_times_Lz, nan=0.0)
    return V_p * volume_average(q_rad, rho, weight=w_V)


_KEV_TO_EV = 1.0e3


def _mavrin_coronal_Lz(symbol: str, Te_keV: np.ndarray) -> np.ndarray:
    """Mavrin 2018 coronal Lz [W*m^3] for one species over a keV grid."""
    entry = _load_radiation_dataset("polynomialfit", "mavrin_coronal", symbol)
    Te = np.clip(Te_keV, _MAVRIN_T_MIN, _MAVRIN_T_MAX)
    return 10.0 ** _binned_log10_Lz(Te, entry["temperature_bin_borders"], entry["radc"])


def mavrin_mean_charge(symbol: str, Te_keV: Any) -> np.ndarray:
    """Mavrin 2018 coronal mean charge Zbar(T_e) for one species."""
    entry = _load_radiation_dataset("polynomialfit", "mavrin_coronal", symbol)
    bins = entry["charge_temperature_bin_borders"]
    zc = entry["zc"]
    Te = np.clip(np.asarray(Te_keV, dtype=float), _MAVRIN_T_MIN, np.nextafter(_MAVRIN_T_MAX, 0.0))
    Tlog = np.log10(Te)
    out = np.zeros_like(Te)
    for i, row in enumerate(zc):
        mask = (Te >= bins[i]) & (Te < bins[i + 1])
        if np.any(mask):
            out[mask] = polyval(Tlog[mask], row)
    return np.clip(out, 0.0, float(entry["atomic_number"]))


def _cooling_integral(symbol: str, target_electron_temp: Any, T_sep: Any) -> Any:
    """L_int = int Lz(Te)*sqrt(Te) dTe in the eV-based Lengyel convention."""
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
    """Edge impurity concentration to cool the SOL (Lengyel model)."""
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
    """Edge impurity concentration referred to the core."""
    # CHECK
    return edge_impurity_concentration / edge_impurity_enrichment
