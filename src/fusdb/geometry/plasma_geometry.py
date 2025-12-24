"""Plasma geometry relations defined once and solved generically."""

import math

from fusdb.relations_values import PRIORITY_RELATION, PRIORITY_STRICT, Relation
from fusdb.relations_util import require_nonzero


def plasma_volume(a: float, R: float, kappa: float, delta: float, xi: float) -> float:
    epsilon = a / R
    theta07 = math.asin(0.7) if xi == 0 else math.asin(0.7) + (1 - math.sqrt(1 + 8 * xi**2)) / (4 * xi)
    w_07 = math.cos(theta07 - xi * math.sin(2 * theta07)) / math.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    S_phi = math.pi * a**2 * kappa * (1 + 0.52 * (w_07 - 1))
    return 2 * math.pi * R * (1 - 0.25 * delta * epsilon) * S_phi


def plasma_surface_area(a: float, R: float, kappa: float, delta: float, xi: float) -> float:
    epsilon = a / R
    theta07 = math.asin(0.7) if xi == 0 else math.asin(0.7) + (1 - math.sqrt(1 + 8 * xi**2)) / (4 * xi)
    w_07 = math.cos(theta07 - xi * math.sin(2 * theta07)) / math.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    Lp = 2 * math.pi * a * (1 + 0.55 * (kappa - 1)) * (1 + 0.08 * delta**2) * (1 + 0.2 * (w_07 - 1))
    return 2 * math.pi * R * (1 - 0.32 * delta * epsilon) * Lp


GEOMETRY_RELATIONS: tuple[Relation, ...] = (
    Relation(
        "Major radius",
        ("R", "R_max", "R_min"),
        lambda v: v["R"] - (v["R_max"] + v["R_min"]) / 2,
        priority=PRIORITY_STRICT,
        initial_guesses={
            "R": lambda v: (v["R_max"] + v["R_min"]) / 2,
            "R_max": lambda v: 2 * v["R"] - v["R_min"],
            "R_min": lambda v: 2 * v["R"] - v["R_max"],
        },
    ),
    Relation(
        "Aspect ratio",
        ("A", "R", "a"),
        lambda v: (require_nonzero(v["a"], "a", "geometry relations") or v["A"] - v["R"] / v["a"]),
        priority=PRIORITY_STRICT,
        initial_guesses={
            "A": lambda v: v["R"] / v["a"],
            "R": lambda v: v["A"] * v["a"],
            "a": lambda v: v["R"] / v["A"],
        },
    ),
    Relation(
        "Elongation",
        ("kappa", "Z_max", "Z_min", "R_max", "R_min"),
        lambda v: (
            require_nonzero(v["R_max"] - v["R_min"], "R_span", "geometry relations")
            or v["kappa"] - (v["Z_max"] - v["Z_min"]) / (v["R_max"] - v["R_min"])
        ),
        priority=PRIORITY_RELATION,
        initial_guesses={
            "kappa": lambda v: (v["Z_max"] - v["Z_min"]) / (v["R_max"] - v["R_min"]),
            "Z_max": lambda v: v["Z_min"] + v["kappa"] * (v["R_max"] - v["R_min"]),
            "Z_min": lambda v: v["Z_max"] - v["kappa"] * (v["R_max"] - v["R_min"]),
            "R_max": lambda v: v["R_min"] + (v["Z_max"] - v["Z_min"]) / v["kappa"],
            "R_min": lambda v: v["R_max"] - (v["Z_max"] - v["Z_min"]) / v["kappa"],
        },
    ),
)

TOKAMAK_SHAPE_RELATIONS: tuple[Relation, ...] = (
    Relation(
        "Elongation 95%",
        ("kappa", "kappa_95"),
        lambda v: v["kappa"] - 1.12 * v["kappa_95"],  # ITER Physics Design Guidelines (Uckan et al. 1990)
        priority=PRIORITY_RELATION,
        initial_guesses={"kappa": lambda v: 1.12 * v["kappa_95"], "kappa_95": lambda v: v["kappa"] / 1.12},
    ),
    Relation(
        "Triangularity 95%",
        ("delta", "delta_95"),
        lambda v: v["delta"] - 1.5 * v["delta_95"],  # ITER Physics Design Guidelines (Uckan et al. 1990)
        priority=PRIORITY_RELATION,
        initial_guesses={"delta": lambda v: 1.5 * v["delta_95"], "delta_95": lambda v: v["delta"] / 1.5},
    ),
    Relation(
        "Tokamak volume",
        ("V_p", "a", "R", "kappa", "delta_95", "squareness"),
        lambda v: v["V_p"] - plasma_volume(v["a"], v["R"], v["kappa"], v["delta_95"], v["squareness"]),
        priority=PRIORITY_STRICT,
        solve_for=("V_p", "a", "R"),
        initial_guesses={
            "V_p": lambda v: plasma_volume(v["a"], v["R"], v["kappa"], v["delta_95"], v["squareness"]),
            "a": lambda v: max(1e-3, (abs(v.get("V_p", 1.0)) ** (1 / 3)) / max(v.get("kappa", 1.0), 1e-3)),
            "R": lambda v: max(1e-3, (abs(v.get("V_p", 1.0)) ** (1 / 3)) * max(v.get("kappa", 1.0), 1e-3)),
        },
    ),
    Relation(
        "Tokamak surface",
        ("S_p", "a", "R", "kappa", "delta_95", "squareness"),
        lambda v: v["S_p"] - plasma_surface_area(v["a"], v["R"], v["kappa"], v["delta_95"], v["squareness"]),
        priority=PRIORITY_STRICT,
        solve_for=("S_p", "a", "R"),
        initial_guesses={
            "S_p": lambda v: plasma_surface_area(v["a"], v["R"], v["kappa"], v["delta_95"], v["squareness"]),
            "a": lambda v: max(1e-3, (abs(v.get("S_p", 1.0)) ** (1 / 2)) / max(v.get("kappa", 1.0), 1e-3)),
            "R": lambda v: max(1e-3, (abs(v.get("S_p", 1.0)) ** (1 / 2))),
        },
    ),
)

# Configuration-specific geometry guidance (simplified from PROCESS/STAR/ITER sources).
# For spherical tokamaks, see Menard et al., Nucl. Fusion 2016 and PROCESS Issue #1439/#1086.
SPHERICAL_TOKAMAK_SHAPE_RELATIONS: tuple[Relation, ...] = (
    Relation(
        "ST elongation vs aspect ratio",
        ("kappa", "A"),
        lambda v: v["kappa"] - 0.95 * (1.9 + 1.9 / (v["A"] ** 1.4)),
        priority=PRIORITY_RELATION,
        solve_for=("kappa",),
        initial_guesses={"kappa": lambda v: 2.0},
    ),
    Relation(
        "ST triangularity vs aspect ratio",
        ("delta_95", "A"),
        lambda v: v["delta_95"] - 0.53 * (1 + 0.77 * (1 / v["A"]) ** 3) / 1.50,
        priority=PRIORITY_RELATION,
        solve_for=("delta_95",),
        initial_guesses={"delta_95": lambda v: 0.3},
    ),
)

STELLARATOR_SHAPE_RELATIONS: tuple[Relation, ...] = (
    
    
)
FRC_SHAPE_RELATIONS: tuple[Relation, ...] = (
    
    
)
MIRROR_SHAPE_RELATIONS: tuple[Relation, ...] = (
    
    
)
