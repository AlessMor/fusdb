"""Shared fusdb-vs-PROCESS comparison surface.

One place defines *what* is compared and *against which MFILE key*, so the three
test modules and the three notebooks cannot drift apart.
"""

from __future__ import annotations

import numpy as np
from fusdb.utils import volume_average

from _process_fixture import build_reactor
from _process_mfile import read_mfile

MW = 1.0e6

# fusdb name -> (MFILE key, scale applied to the PROCESS value, short label).
#
# Two mappings here are easy to get wrong and were:
#   * `W_th` is compared against `e_plasma_beta_thermal`, NOT `e_plasma_beta` --
#     the latter includes fast-alpha beta (~13% of the total here).
#   * `qstar` is compared against PROCESS's `q95`: fusdb's own `q95` variable has
#     no producer and is input-only, while its "Edge safety factor q_star" IS
#     PROCESS's q95 formula.
FIELDS: dict[str, tuple[str, float, str]] = {
    # geometry
    "V_p": ("vol_plasma", 1.0, "plasma volume [m3]"),
    "A_p": ("a_plasma_surface", 1.0, "plasma surface area [m2]"),
    "S_phi": ("a_plasma_poloidal", 1.0, "poloidal cross-section [m2]"),
    # profiles / composition
    "n_e_avg": ("nd_plasma_electrons_vol_avg", 1.0, "volume-avg electron density [m-3]"),
    "n_la": ("nd_plasma_electron_line", 1.0, "line-avg electron density [m-3]"),
    "Z_eff": ("n_charge_plasma_effective_vol_avg", 1.0, "effective charge [-]"),
    # beta / stored energy
    "beta": ("beta_thermal_vol_avg", 1.0, "thermal beta [-]"),
    "W_th": ("e_plasma_beta_thermal", 1.0, "thermal stored energy [J]"),
    # current
    "f_BS": ("f_c_plasma_bootstrap", 1.0, "bootstrap fraction [-]"),
    "qstar": ("q95", 1.0, "edge safety factor [-]"),
    # power balance
    "P_fus": ("p_fusion_total_mw", MW, "fusion power [W]"),
    "P_neutron": ("p_neutron_total_mw", MW, "neutron power [W]"),
    "P_rad": ("p_plasma_rad_mw", MW, "total radiated power [W]"),
    "P_rad_core": ("p_plasma_inner_rad_mw", MW, "core radiated power [W]"),
    "P_sync": ("p_plasma_sync_mw", MW, "synchrotron power [W]"),
    "P_loss": ("p_plasma_loss_mw", MW, "transport loss power [W]"),
    "P_sep": ("p_plasma_separatrix_mw", MW, "power crossing separatrix [W]"),
    "P_LH": ("p_l_h_threshold_mw", MW, "L-H threshold power [W]"),
    "P_aux": ("p_hcd_injected_total_mw", MW, "injected auxiliary power [W]"),
    # confinement
    "tau_E": ("t_energy_confinement", 1.0, "energy confinement time [s]"),
    "Q_sci": ("big_q_plasma", 1.0, "fusion gain [-]"),
}

# Quantities that need a small expression on one or both sides, because the two
# codes hold them with different denominators or different inclusions.  These are
# ordinary unit conversions, but they are COMPARED rather than merely asserted in
# prose -- the composition conversion in _process_fixture.py is exactly the kind
# of thing that is easy to get silently wrong.
#
# fusdb name -> (fusdb expression, PROCESS expression, label)
DERIVED: dict[str, tuple] = {
    "n_fuel_avg": (
        lambda v: vol_avg(v, "n_D") + vol_avg(v, "n_T"),
        lambda d: d["nd_plasma_fuel_ions_vol_avg"][1],
        "volume-avg FUEL ion density [m-3] (fusdb n_D+n_T)",
    ),
    "n_He4_avg": (
        lambda v: vol_avg(v, "n_He4"),
        # PROCESS quotes the ash fraction against the ELECTRON density, whereas
        # fusdb's f_He4 is against the total ION density.
        lambda d: d["f_nd_alpha_thermal_electron"][1] * d["nd_plasma_electrons_vol_avg"][1],
        "volume-avg helium ash density [m-3]",
    ),
    "W_total": (
        # fusdb has no fast-alpha pressure, so its stored energy is thermal only.
        # Compared against PROCESS's THERMAL part, not its e_plasma_beta, which
        # includes ~14% fast-alpha beta.
        lambda v: scalar(v, "W_th"),
        lambda d: d["e_plasma_beta_thermal"][1],
        "thermal stored energy [J] (vs PROCESS thermal, not total)",
    ),
}

# Every name the tests assert on: the direct MFILE mappings plus the derived
# conversions above.
COMPARED = (*FIELDS, *DERIVED)

TOLERANCE = 0.10


def scalar(values, name):
    """Read one scalar out of a fusdb result value map."""
    if name not in values:
        return None
    raw = values[name]
    return float(np.asarray(raw).ravel()[0]) if np.ndim(raw) else float(raw)


def vol_avg(values, name):
    """Flux-volume average of a profile in a fusdb result value map.

    fusdb stores the composition as profiles (``n_D``, ``n_T``, ``n_He4``) and
    exposes rho-averages, not volume-averages; PROCESS reports volume-averages,
    so the reduction has to be done here to compare like with like.
    """
    if name not in values or "rho" not in values:
        return None
    return float(volume_average(np.asarray(values[name], dtype=float),
                                np.asarray(values["rho"], dtype=float)))


def solve(mfile_path):
    """Reconcile the fixture for one PROCESS run; return (values, result)."""
    result = build_reactor(mfile_path).reconcile()
    return result["values"], result


def compare(mfile_path):
    """Return {fusdb_name: {fusdb, process, rel_error, label}} for one run."""
    values, result = solve(mfile_path)
    data = read_mfile(mfile_path)
    out = {}
    for name, (key, scale, label) in FIELDS.items():
        got = scalar(values, name)
        entry = data.get(key)
        if got is None or entry is None:
            continue
        reference = entry[1] * scale
        out[name] = {
            "fusdb": got,
            "process": reference,
            "rel_error": (got - reference) / reference if reference else float("nan"),
            "label": label,
        }
    for name, (fusdb_expr, process_expr, label) in DERIVED.items():
        try:
            got, reference = fusdb_expr(values), process_expr(data)
        except (KeyError, TypeError):
            continue
        if got is None or reference in (None, 0):
            continue
        out[name] = {
            "fusdb": got,
            "process": reference,
            "rel_error": (got - reference) / reference,
            "label": label,
        }
    return out, result
