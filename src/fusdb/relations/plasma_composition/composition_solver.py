"""Residual-based steady-state plasma composition solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping

import numpy as np

from fusdb.relations.reactivities.reactivity_functions import (
    sigmav_DDn_BoschHale,
    sigmav_DDp_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_He3He3_CF88,
    sigmav_THe3_D_CF88,
    sigmav_THe3_np_CF88,
    sigmav_TT_CF88,
)


_TRACKED_SPECIES = ("D", "T", "He3", "He4")
_REDUCED_SPECIES = ("T", "He3", "He4")
_FRACTION_VARS = {"D": "f_D", "T": "f_T", "He3": "f_He3", "He4": "f_He4"}
_IMPURITY_KEYS = ("f_imp", "impurity", "f_impurity")
_REACTION_REACTANTS = {
    "DT": ("D", "T"),
    "DDn": ("D", "D"),
    "DDp": ("D", "D"),
    "DHe3": ("D", "He3"),
    "TT": ("T", "T"),
    "He3He3": ("He3", "He3"),
    "THe3_D": ("T", "He3"),
    "THe3_np": ("T", "He3"),
}
_DEFAULT_REACTIVITY_FUNCTIONS = {
    "DT": sigmav_DT_BoschHale,
    "DDn": sigmav_DDn_BoschHale,
    "DDp": sigmav_DDp_BoschHale,
    "DHe3": sigmav_DHe3_BoschHale,
    "TT": sigmav_TT_CF88,
    "He3He3": sigmav_He3He3_CF88,
    "THe3_D": sigmav_THe3_D_CF88,
    "THe3_np": sigmav_THe3_np_CF88,
}


def _finite_scalar(value: object, *, name: str, allow_zero: bool = True, positive: bool = False) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0:
        raise ValueError(f"{name} must be > 0")
    if not allow_zero and scalar == 0:
        raise ValueError(f"{name} must be non-zero")
    return scalar


def _extract_fraction_inputs(mapping: Mapping[str, float] | None) -> tuple[dict[str, float], float | None]:
    raw: dict[str, float] = {}
    impurity = None
    if mapping is None:
        return raw, impurity

    for species in _TRACKED_SPECIES:
        for key in (_FRACTION_VARS[species], species):
            if key in mapping:
                raw[species] = _finite_scalar(mapping[key], name=_FRACTION_VARS[species])
                break

    for key in _IMPURITY_KEYS:
        if key in mapping:
            impurity = _finite_scalar(mapping[key], name="f_imp")
            break
    return raw, impurity


def _normalize_weights(raw: Mapping[str, float], *, total: float, label: str) -> dict[str, float]:
    if total < 0:
        raise ValueError("tracked total fraction must be >= 0")
    if total == 0:
        return {species: 0.0 for species in _TRACKED_SPECIES}

    weights = {species: float(raw.get(species, 0.0)) for species in _TRACKED_SPECIES}
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError(f"{label} cannot contain negative fractions")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError(f"{label} must contain at least one positive tracked fraction")
    return {
        species: total * weight / weight_sum
        for species, weight in weights.items()
    }


@dataclass(slots=True)
class CompositionSteadyStateSystem:
    """Fast steady-state composition model with a SciPy root solve.

    The class is steady-state first: it precomputes the reaction coefficients and
    solves the particle balance residual directly. The same balance is exposed as
    ``rhs`` so the model can later be reused with ``solve_ivp`` if needed.
    """

    n_i: float
    T_avg: float
    fractions: Mapping[str, float] | None = None
    initial_guess: Mapping[str, float] | None = None
    tau_p: Mapping[str, float] | None = None
    temperatures: Mapping[str, float] | None = None
    source_distribution: Mapping[str, float] | None = None
    f_imp: float | None = None
    reactivity_functions: Mapping[str, Callable[[float], float]] | None = None

    tracked_total: float = field(init=False)
    initial_fractions: dict[str, float] = field(init=False)
    source_fractions: dict[str, float] = field(init=False)
    tau_p_map: dict[str, float] = field(init=False)
    reactivity_values: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.n_i = _finite_scalar(self.n_i, name="n_i", positive=True)
        self.T_avg = _finite_scalar(self.T_avg, name="T_avg")
        if self.T_avg < 0.0:
            raise ValueError("T_avg must be >= 0")

        fraction_inputs, impurity_from_fractions = _extract_fraction_inputs(self.fractions)
        guess_inputs, impurity_from_guess = _extract_fraction_inputs(self.initial_guess)
        source_inputs, impurity_from_source = _extract_fraction_inputs(self.source_distribution)

        impurity = self.f_imp
        if impurity is None:
            if impurity_from_fractions is not None:
                impurity = impurity_from_fractions
            elif impurity_from_guess is not None:
                impurity = impurity_from_guess
            else:
                impurity = impurity_from_source
        if impurity is None:
            impurity = 0.0
        impurity = _finite_scalar(impurity, name="f_imp")
        if impurity < 0 or impurity >= 1:
            raise ValueError("f_imp must satisfy 0 <= f_imp < 1")
        self.f_imp = impurity
        self.tracked_total = 1.0 - impurity

        if not fraction_inputs:
            fraction_inputs = {"D": 0.5, "T": 0.5}
        if not guess_inputs:
            guess_inputs = dict(fraction_inputs)
        if not source_inputs:
            source_inputs = dict(fraction_inputs)

        self.initial_fractions = _normalize_weights(
            guess_inputs,
            total=self.tracked_total,
            label="initial_guess",
        )

        self.source_fractions = _normalize_weights(
            source_inputs,
            total=1.0,
            label="source_distribution",
        )

        self.tau_p_map = {}
        for species in _TRACKED_SPECIES:
            tau_value = None
            if self.tau_p is not None:
                for key in (species, f"tau_p_{species}", "tau_p", "default"):
                    if key in self.tau_p:
                        tau_value = self.tau_p[key]
                        break
            if tau_value is None:
                raise ValueError(f"tau_p_{species} is required")
            self.tau_p_map[species] = _finite_scalar(tau_value, name=f"tau_p_{species}", positive=True)

        funcs = dict(_DEFAULT_REACTIVITY_FUNCTIONS)
        if self.reactivity_functions:
            funcs.update(self.reactivity_functions)
        self.reactivity_values = {
            reaction: self._evaluate_reactivity(
                reaction,
                func,
                self._reaction_temperature(reaction),
            )
            for reaction, func in funcs.items()
        }

    def _evaluate_reactivity(self, reaction: str, func: Callable[[float], float], temperature: float) -> float:
        if temperature < 0.0:
            raise ValueError(f"{reaction} temperature must be >= 0")
        if temperature == 0.0:
            return 0.0
        try:
            value = float(func(temperature))
        except ZeroDivisionError:
            return 0.0
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{reaction} reactivity must be finite and >= 0")
        return value

    def _species_temperature(self, species: str) -> float:
        if self.temperatures is not None:
            for key in (species, f"T_{species}", f"T_i_{species}"):
                if key in self.temperatures:
                    value = _finite_scalar(self.temperatures[key], name=f"T_i_{species}")
                    if value < 0.0:
                        raise ValueError(f"T_i_{species} must be >= 0")
                    return value
        return self.T_avg

    def _reaction_temperature(self, reaction: str) -> float:
        if self.temperatures is not None:
            for key in (reaction, f"T_{reaction}", reaction.lower()):
                if key in self.temperatures:
                    value = _finite_scalar(self.temperatures[key], name=f"T_{reaction}")
                    if value < 0.0:
                        raise ValueError(f"T_{reaction} must be >= 0")
                    return value

        reactants = _REACTION_REACTANTS[reaction]
        return sum(self._species_temperature(species) for species in reactants) / len(reactants)

    def initial_state(self) -> np.ndarray:
        """Return the full tracked-fraction state used as the default root initial guess."""
        return np.asarray([self.initial_fractions[species] for species in _TRACKED_SPECIES], dtype=float)

    def _full_state_from_reduced(self, reduced: np.ndarray) -> np.ndarray | None:
        if reduced.shape != (len(_REDUCED_SPECIES),):
            raise ValueError(f"reduced state must have shape {(len(_REDUCED_SPECIES),)}")
        f_T, f_He3, f_He4 = (float(value) for value in reduced)
        f_D = self.tracked_total - f_T - f_He3 - f_He4
        full = np.asarray([f_D, f_T, f_He3, f_He4], dtype=float)
        if np.any(~np.isfinite(full)):
            return None
        if np.any(full < 0.0):
            return None
        return full

    def _fraction_map(self, state: np.ndarray) -> dict[str, float]:
        return {species: float(value) for species, value in zip(_TRACKED_SPECIES, state)}

    def _reaction_fraction_rates(self, state: np.ndarray) -> dict[str, float]:
        f_D, f_T, f_He3, _f_He4 = (float(value) for value in state)
        n_scale = self.n_i
        return {
            "DT": n_scale * f_D * f_T * self.reactivity_values["DT"],
            "DDn": 0.5 * n_scale * f_D * f_D * self.reactivity_values["DDn"],
            "DDp": 0.5 * n_scale * f_D * f_D * self.reactivity_values["DDp"],
            "DHe3": n_scale * f_D * f_He3 * self.reactivity_values["DHe3"],
            "TT": 0.5 * n_scale * f_T * f_T * self.reactivity_values["TT"],
            "He3He3": 0.5 * n_scale * f_He3 * f_He3 * self.reactivity_values["He3He3"],
            "THe3_D": n_scale * f_T * f_He3 * self.reactivity_values["THe3_D"],
            "THe3_np": n_scale * f_T * f_He3 * self.reactivity_values["THe3_np"],
        }

    def rhs(self, _t: float, state: np.ndarray) -> np.ndarray:
        """Return full tracked-species fraction derivatives with total tracked fraction preserved."""
        state = np.asarray(state, dtype=float)
        if state.shape != (len(_TRACKED_SPECIES),):
            raise ValueError(f"state must have shape {(len(_TRACKED_SPECIES),)}")

        rates = self._reaction_fraction_rates(state)
        f_D, f_T, f_He3, f_He4 = (float(value) for value in state)

        raw = np.asarray(
            [
                (
                    -rates["DT"]
                    - 2.0 * rates["DDn"]
                    - 2.0 * rates["DDp"]
                    - rates["DHe3"]
                    + rates["THe3_D"]
                    - f_D / self.tau_p_map["D"]
                ),
                (
                    +rates["DDp"]
                    - rates["DT"]
                    - 2.0 * rates["TT"]
                    - rates["THe3_D"]
                    - rates["THe3_np"]
                    - f_T / self.tau_p_map["T"]
                ),
                (
                    +rates["DDn"]
                    - rates["DHe3"]
                    - 2.0 * rates["He3He3"]
                    - rates["THe3_D"]
                    - rates["THe3_np"]
                    - f_He3 / self.tau_p_map["He3"]
                ),
                (
                    +rates["DT"]
                    + rates["DHe3"]
                    + rates["TT"]
                    + rates["He3He3"]
                    + rates["THe3_D"]
                    + rates["THe3_np"]
                    - f_He4 / self.tau_p_map["He4"]
                ),
            ],
            dtype=float,
        )

        refill_rate = -float(np.sum(raw))
        refill = refill_rate * np.asarray(
            [self.source_fractions[species] for species in _TRACKED_SPECIES],
            dtype=float,
        )
        return raw + refill

    def residual(self, reduced: np.ndarray) -> np.ndarray:
        """Return the reduced steady-state residual used by scipy.optimize.root."""
        reduced = np.asarray(reduced, dtype=float)
        state = self._full_state_from_reduced(reduced)
        if state is None:
            penalty = 1e3 + np.sum(np.abs(reduced))
            return np.full(len(_REDUCED_SPECIES), penalty, dtype=float)
        return self.rhs(0.0, state)[1:]

    def solve(self, *, tol: float = 1e-10, maxfev: int = 500, method: str = "hybr") -> dict[str, float]:
        """Solve the steady-state composition and return tracked fractions."""
        try:
            from scipy.optimize import root
        except Exception as exc:
            raise ImportError("solve_steady_state_composition requires scipy.optimize.root") from exc

        initial = self.initial_state()[1:]
        result = root(
            self.residual,
            initial,
            method=method,
            tol=tol,
            options={"maxfev": maxfev},
        )
        if not result.success:
            residual_norm = float(np.linalg.norm(self.residual(result.x)))
            raise RuntimeError(
                f"Steady-state composition solve failed: {result.message} (||res||={residual_norm:.3e})"
            )

        state = self._full_state_from_reduced(np.asarray(result.x, dtype=float))
        if state is None:
            raise RuntimeError("Steady-state composition solve produced an invalid state")

        residual_norm = float(np.linalg.norm(self.rhs(0.0, state)[1:]))
        if not math.isfinite(residual_norm) or residual_norm > max(tol, 1e-12):
            raise RuntimeError(
                f"Steady-state composition solve did not converge tightly enough (||res||={residual_norm:.3e})"
            )

        return {
            _FRACTION_VARS[species]: float(value)
            for species, value in self._fraction_map(state).items()
        }


def solve_steady_state_composition(
    n_i: float,
    T_avg: float,
    fractions: Mapping[str, float] | None = None,
    tau_p: Mapping[str, float] | None = None,
    *,
    initial_guess: Mapping[str, float] | None = None,
    temperatures: Mapping[str, float] | None = None,
    source_distribution: Mapping[str, float] | None = None,
    f_imp: float | None = None,
    reactivity_functions: Mapping[str, Callable[[float], float]] | None = None,
    tol: float = 1e-10,
    max_iter: int = 500,
    method: str = "hybr",
) -> dict[str, float]:
    """Solve for the steady-state tracked ion fractions.

    The tracked species fractions are constrained to sum to ``1 - f_imp``.
    ``source_distribution`` defines the external particle source mix used to keep
    the tracked fraction total constant during the steady-state balance.
    ``initial_guess`` controls only the nonlinear solver start point. ``fractions``
    is retained as a shorthand that seeds both when the explicit arguments are not
    provided.
    """
    system = CompositionSteadyStateSystem(
        n_i=n_i,
        T_avg=T_avg,
        fractions=fractions,
        initial_guess=initial_guess,
        tau_p=tau_p,
        temperatures=temperatures,
        source_distribution=source_distribution,
        f_imp=f_imp,
        reactivity_functions=reactivity_functions,
    )
    return system.solve(tol=tol, maxfev=max_iter, method=method)
