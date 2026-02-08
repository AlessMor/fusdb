"""Ion composition relations generalized by allowed species."""
from __future__ import annotations
from inspect import Signature, Parameter
from pathlib import Path
from fusdb.relation_class import Relation_decorator as Relation
from fusdb.utils import load_yaml

_REGISTRY = Path(__file__).resolve().parents[2] / "registry" / "allowed_species.yaml"

def _load_species() -> dict:
    """Load allowed species metadata. Args: none. Returns: dict."""
    try:
        return load_yaml(_REGISTRY)
    except Exception:
        return {}

_SPECIES_DATA = _load_species()
SPECIES = tuple(_SPECIES_DATA.keys())
FRACTIONS = [f"f_{s}" for s in SPECIES]
DENSITIES = {s: f"n_{s}" for s in SPECIES}
CHARGE = [(_SPECIES_DATA.get(s, {}) or {}).get("atomic_number", 1) for s in SPECIES]
MASS = [(_SPECIES_DATA.get(s, {}) or {}).get("atomic_mass", 1) for s in SPECIES]


def _relation(name: str, output: str, inputs: list[str], func, constraints=None, rel_tol=None):
    """Create a Relation with standard plasma tags. Args: name, output, inputs, func. Returns: Relation."""
    func.__signature__ = Signature([Parameter(n, Parameter.POSITIONAL_OR_KEYWORD) for n in inputs])
    return Relation(name=name, output=output, tags=("plasma",), constraints=constraints, rel_tol_default=rel_tol)(func)


def _fraction_sum_expr() -> str:
    """Build sum(f_i) constraint string. Args: none. Returns: str."""
    return " + ".join(FRACTIONS) if FRACTIONS else "0"


def _charge_sum_expr() -> str:
    """Build sum(Z_i * f_i) constraint string. Args: none. Returns: str."""
    terms = [f"{z}*{f}" if z != 1 else f for z, f in zip(CHARGE, FRACTIONS)]
    return " + ".join(terms) if terms else "0"


# Dynamic fraction relations.
_relations = []

if FRACTIONS:
    sum_expr = _fraction_sum_expr()
    charge_expr = _charge_sum_expr()

    def _ion_density(*args):
        n_e = args[0]
        denom = sum(val * w for val, w in zip(args[1:], CHARGE))
        return n_e / denom

    _relations.append(
        _relation(
            "Ion density from electron density and fractions",
            "n_i",
            ["n_e", *FRACTIONS],
            _ion_density,
            constraints=(f"{charge_expr} != 0",),
        )
    )

    def _electron_density(*args):
        n_i = args[0]
        denom = sum(val * w for val, w in zip(args[1:], CHARGE))
        return n_i * denom

    _relations.append(
        _relation(
            "Electron density from ion fractions",
            "n_e",
            ["n_i", *FRACTIONS],
            _electron_density,
        )
    )

    def _fraction_check(*args):
        return args[0]

    _relations.append(
        _relation(
            "Fuel fraction sum",
            FRACTIONS[0],
            FRACTIONS,
            _fraction_check,
            constraints=(f"Abs({sum_expr} - 1) <= 1e-6",),
            rel_tol=1e-6,
        )
    )

    for idx, frac in enumerate(FRACTIONS):
        def _fraction_norm(*args, _i=idx):
            total = sum(args)
            if total <= 0:
                return 1.0 / len(args)
            return args[_i] / total
        _relations.append(
            _relation(
                f"Fuel fraction equilibrium {SPECIES[idx]}",
                frac,
                FRACTIONS,
                _fraction_norm,
                constraints=(f"Abs({sum_expr} - 1) <= 1e-6",),
                rel_tol=1e-6,
            )
        )

    for idx, species in enumerate(SPECIES):
        f_var = FRACTIONS[idx]
        n_var = DENSITIES[species]

        def _dens(*args):
            return args[0] * args[1]
        _relations.append(
            _relation(
                f"{species} density from fraction",
                n_var,
                [f_var, "n_i"],
                _dens,
            )
        )

        def _frac(*args):
            return args[0] / args[1]
        _relations.append(
            _relation(
                f"{species} fraction from density",
                f_var,
                [n_var, "n_i"],
                _frac,
                constraints=("n_i > 0",),
            )
        )

        if len(FRACTIONS) > 1:
            others = [f for f in FRACTIONS if f != f_var]
            def _norm_other(*args):
                return 1.0 - sum(args)
            _relations.append(
                _relation(
                    f"Ion fraction normalization (solve {f_var})",
                    f_var,
                    others,
                    _norm_other,
                )
            )

    def _afuel(*args):
        return sum(val * w for val, w in zip(args, MASS))
    _relations.append(
        _relation(
            "Average fuel mass number",
            "afuel",
            FRACTIONS,
            _afuel,
        )
    )

