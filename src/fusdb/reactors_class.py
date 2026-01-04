from dataclasses import dataclass, field, fields as dataclass_fields
import importlib
import inspect
import math
from pathlib import Path
from typing import Any, ClassVar

import warnings

from fusdb.power_exhaust.power_exhaust import PSEP_RELATIONS
from fusdb.relation_class import PRIORITY_RELATION, Relation, RelationSystem
from fusdb.relations_util import REL_TOL_DEFAULT


@dataclass
class Reactor:
    ALLOWED_REACTOR_CLASSES: ClassVar[tuple[str, ...]] = (
        "ARC-class",
        "ARC",
        "STEP-class",
        "STEP",
        "DEMO-class",
        "DEMO",
    )
    ALLOWED_REACTOR_CONFIGURATIONS: ClassVar[tuple[str, ...]] = (
        "tokamak",
        "compact tokamak",
        "spherical tokamak",
        "stellarator",
    )
    _RELATION_MODULES: ClassVar[tuple[str, ...]] = (
        "fusdb.geometry.plasma_geometry",
        "fusdb.plasma_pressure.beta",
        "fusdb.confinement.plasma_stored_energy",
        "fusdb.confinement.scalings",
    )
    _RELATIONS: ClassVar[list[tuple[tuple[str, ...], Relation]]] = []
    _RELATIONS_IMPORTED: ClassVar[bool] = False
    ALLOWED_VARS: ClassVar[set[str]] = set()

    # required identity
    id: str = field(metadata={"section": "metadata_required"})
    name: str = field(metadata={"section": "metadata_required"})
    reactor_configuration: str = field(metadata={"section": "metadata_required"})
    organization: str = field(metadata={"section": "metadata_required"})

    # optional general/scenario metadata
    reactor_class: str | None = field(default=None, metadata={"section": "metadata_optional"})
    country: str | None = field(default=None, metadata={"section": "metadata_optional"})
    site: str | None = field(default=None, metadata={"section": "metadata_optional"})
    design_year: int | None = field(default=None, metadata={"section": "metadata_optional"})
    doi: str | list[str] | None = field(default=None, metadata={"section": "metadata_optional"})
    notes: str | None = field(default=None, metadata={"section": "metadata_optional"})
    allow_relation_overrides: bool | None = field(default=False, metadata={"section": "metadata_optional"})

    # plasma geometry
    R: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "m"})  # major radius
    a: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "m"})  # minor radius
    A: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "1"})  # aspect ratio
    kappa: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "1"})  # elongation κ
    kappa_95: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "1"})  # elongation at 95%
    delta_95: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "1"})  # triangularity δ95
    delta: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "1"})  # triangularity core
    V_p: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "m^3"})  # plasma volume
    S_p: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "m^2"})  # plasma surface area
    squareness: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "1"})  # squareness ξ
    R_max: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "m"})  # outer major radius
    R_min: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "m"})  # inner major radius
    Z_max: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "m"})  # top vertical extent
    Z_min: float | None = field(default=None, metadata={"section": "plasma_geometry", "unit": "m"})  # bottom vertical extent

    # plasma parameters
    B0: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "T"})  # toroidal field on axis
    B_max: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "T"})  # peak field on coil
    B_p: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "T"})  # poloidal field
    I_p: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "MA"})  # plasma current
    f_BS: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # bootstrap fraction
    f_NI: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # non-inductive fraction
    f_CD: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # driven current fraction
    q95: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # q at r/a = 0.95
    q_a: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # edge q
    q_min: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # minimum q
    li: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # internal inductance
    beta_N: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # normalized beta βN
    beta_T: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # toroidal beta (fraction)
    beta_p: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # poloidal beta
    beta: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # total beta
    T_avg: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "keV"})  # ⟨T⟩
    T0: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "keV"})  # T0
    T_e: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "keV"})  # electron temperature
    T_i: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "keV"})  # ion temperature
    n_avg: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1/m^3"})  # ⟨n⟩
    n0: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1/m^3"})  # n0
    n_e: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1/m^3"})  # electron density
    n_i: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1/m^3"})  # ion density
    f_GW: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # Greenwald fraction
    Z_eff: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # effective charge
    p_th: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "Pa"})  # thermal pressure
    W_th: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "J"})  # thermal stored energy
    P_loss: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "W"})  # power loss for confinement
    TBR: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # tritium breeding ratio
    tau_E: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "s"})  # energy confinement time
    tau_E_method: str | None = field(
        default=None, metadata={"section": "plasma_parameters", "unit": "method"}
    )  # chosen confinement scaling name (e.g., tau_E_mirnov)
    H89: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # H89 confinement factor
    H98_y2: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # H98(y,2) confinement factor
    G89: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "1"})  # G89 gain factor
    tau_pulse: float | None = field(default=None, metadata={"section": "plasma_parameters", "unit": "s"})  # pulse length

    # power and efficiency
    P_fus: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW"})  # fusion power
    P_th_tot: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW"})  # total thermal power
    P_elec_tot: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW"})  # total electric power
    P_elec_net: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW"})  # net electric power
    eta_th: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "1"})  # plant thermal efficiency
    Q_e: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "1"})  # electric power multiplication
    P_wall: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW/m^2"})  # Pf/Sb
    P_LHCD: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW"})  # LHCD coupled
    P_ICRF: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW"})  # ICRF coupled
    P_LH: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW"})  # LH threshold
    P_sep: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW"})  # power across separatrix
    P_sep_over_R: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW/m"})  # Psep/R
    P_sep_B_over_q95AR: float | None = field(default=None, metadata={"section": "power_and_efficiency", "unit": "MW*T/m"})  # Psep*B/(q95*A*R)

    # density profile artifact
    density_profile_file: str | None = field(default=None, metadata={"section": "artifact"})
    density_profile_x_axis: str | None = field(default=None, metadata={"section": "artifact"})
    density_profile_y_dataset: str | None = field(default=None, metadata={"section": "artifact"})
    density_profile_x_unit: str | None = field(default=None, metadata={"section": "artifact"})
    density_profile_y_unit: str | None = field(default=None, metadata={"section": "artifact"})

    # internal
    root_dir: Path | None = field(default=None, metadata={"section": "internal"})
    _sources: dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        type(self)._ensure_allowed_vars()
        if self.allow_relation_overrides is None:
            self.allow_relation_overrides = False
        self.reactor_class = self._normalize_choice(
            self.reactor_class, self.ALLOWED_REACTOR_CLASSES, "reactor_class"
        )
        self.reactor_configuration = self._normalize_choice(
            self.reactor_configuration, self.ALLOWED_REACTOR_CONFIGURATIONS, "reactor_configuration"
        )
        rel_tol = REL_TOL_DEFAULT
        lock = not bool(self.allow_relation_overrides)
        self._solve_geometry(rel_tol, lock)
        self._solve_plasma(rel_tol, lock)
        self._solve_power_exhaust(rel_tol, lock)
        # final source map cleanup
        self._sources = dict(self._sources)

    def _record_sources(self, var_map: dict[str, Any]) -> None:
        for name, var in var_map.items():
            if hasattr(var, "source"):
                self._sources[name] = var.source

    def _is_tokamak_config(self) -> bool:
        concept = (self.reactor_configuration or "").lower()
        return "tokamak" in concept

    def _is_spherical_tokamak(self) -> bool:
        concept = (self.reactor_configuration or "").lower()
        return "spherical tokamak" in concept

    def _is_stellarator_config(self) -> bool:
        concept = (self.reactor_configuration or "").lower()
        return "stellarator" in concept

    def _config_exclude_tags(self) -> tuple[str, ...]:
        if self._is_spherical_tokamak():
            return ("stellarator", "frc", "mirror")
        if self._is_stellarator_config():
            return ("tokamak", "spherical_tokamak", "frc", "mirror")
        if self._is_tokamak_config():
            return ("stellarator", "spherical_tokamak", "frc", "mirror")
        return ()

    def _solve_relations(
        self,
        relations: tuple[Relation, ...],
        rel_tol: float,
        lock: bool,
        seed_overrides: dict[str, float] | None = None,
    ) -> dict[str, float | None]:
        if not relations:
            return {}

        system = RelationSystem(
            [rel.with_tol(rel_tol) for rel in relations], rel_tol=rel_tol, warn=warnings.warn, lock_explicit=lock
        )
        seen_vars: set[str] = set()
        for rel in relations:
            for var in rel.variables:
                if var in seen_vars:
                    continue
                seen_vars.add(var)
                value = getattr(self, var, None) if var in self.ALLOWED_VARS else None
                if value is None and seed_overrides and var in seed_overrides:
                    value = seed_overrides[var]
                system.set(var, value)

        values = system.solve()
        for var in seen_vars:
            if var not in self.ALLOWED_VARS:
                continue
            current = getattr(self, var, None)
            new_value = values.get(var, current)
            if new_value is not None:
                setattr(self, var, new_value)

        self._record_sources(system.var_map)
        return values

    def _solve_geometry(self, rel_tol: float, lock: bool) -> None:
        relations = relations_for(("geometry",), require_all=False, exclude=self._config_exclude_tags())
        seeds = {"delta_95": 0.0, "squareness": 0.0} if self._is_tokamak_config() else None
        self._solve_relations(relations, rel_tol, lock, seed_overrides=seeds)

    def _solve_power_exhaust(self, rel_tol: float, lock: bool) -> None:
        self._solve_relations(tuple(PSEP_RELATIONS), rel_tol, lock)

    def _config_confinement_relations(self) -> tuple[Relation, ...]:
        if self._is_tokamak_config():
            return confinement_relations(("confinement", "tokamak"), require_all=True)
        if self._is_stellarator_config():
            return confinement_relations(("confinement", "stellarator"), require_all=True)
        return ()

    def _select_confinement_relation(self, rel_tol: float) -> Relation | None:
        if self.tau_E is None:
            return None

        candidates = self._config_confinement_relations()
        best: Relation | None = None
        best_diff = math.inf

        for rel in candidates:
            system = RelationSystem([rel.with_tol(rel_tol)], rel_tol=rel_tol, warn=warnings.warn)
            target_var = rel.variables[0]
            for var in rel.variables:
                if var == target_var:
                    continue
                system.set(var, getattr(self, var, None))
            try:
                values = system.solve()
            except Exception:
                continue
            predicted = values.get(target_var)
            if predicted is None:
                continue
            diff = abs(predicted - self.tau_E)
            if diff < best_diff:
                best_diff = diff
                best = rel

        return best

    def _solve_plasma(self, rel_tol: float, lock: bool) -> None:
        relations: list[Relation] = list(relations_for(("plasma",), require_all=True))
        confinement_var = None

        if self.tau_E is not None:
            chosen_rel = self._select_confinement_relation(rel_tol)
            if chosen_rel:
                if self.tau_E_method and self.tau_E_method != chosen_rel.variables[0]:
                    warnings.warn(
                        f"tau_E_method {self.tau_E_method!r} adjusted to {chosen_rel.variables[0]!r} to match tau_E",
                        UserWarning,
                    )
                self.tau_E_method = chosen_rel.variables[0]

        if self.tau_E_method:
            method_rel = confinement_relations_by_name().get(self.tau_E_method)
            if method_rel is None:
                raise ValueError(f"Unknown tau_E_method {self.tau_E_method!r}. Must match a confinement relation name.")
            relations.append(method_rel)
            confinement_var = method_rel.variables[0]
            relations.append(
                Relation(
                    f"Energy confinement method {self.tau_E_method}",
                    ("tau_E", confinement_var),
                    lambda v: v["tau_E"] - v[confinement_var],
                    priority=PRIORITY_RELATION,
                )
            )

        seed_overrides = {confinement_var: self.tau_E} if confinement_var and self.tau_E is not None else None
        self._solve_relations(tuple(relations), rel_tol, lock, seed_overrides=seed_overrides)


    def has_density_profile(self) -> bool:
        return self.density_profile_file is not None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "reactor_class": self.reactor_class,
            "name": self.name,
            "reactor_configuration": self.reactor_configuration,
            "organization": self.organization,
            "country": self.country,
            "design_year": self.design_year,
            "doi": self.doi,
            "P_fus": self.P_fus,
            "R": self.R,
            "n_avg": self.n_avg,
        }

    @staticmethod
    def _normalize_choice(value: str | None, allowed: tuple[str, ...], field_name: str) -> str | None:
        if value is None:
            return None
        mapping = {entry.lower(): entry for entry in allowed}
        key = value.lower()
        if key not in mapping:
            allowed_list = ", ".join(allowed)
            raise ValueError(f"{field_name} must be one of {allowed_list} (case-insensitive); got {value!r}")
        return mapping[key]

    @classmethod
    def _ensure_relation_modules_loaded(cls) -> None:
        if cls._RELATIONS_IMPORTED:
            return
        cls._ensure_allowed_vars()
        for module_name in cls._RELATION_MODULES:
            importlib.import_module(module_name)
        cls._RELATIONS_IMPORTED = True

    @classmethod
    def _ensure_allowed_vars(cls) -> None:
        if cls.ALLOWED_VARS:
            return
        cls.ALLOWED_VARS = {f.name for f in dataclass_fields(cls)}

    @classmethod
    def relation(
        cls,
        groups: str | tuple[str, ...],
        *,
        name: str,
        output: str | None = None,
        variables: tuple[str, ...] | None = None,
        solve_for: tuple[str, ...] | None = None,
        priority: int | None = None,
        rel_tol: float = REL_TOL_DEFAULT,
        initial_guesses: dict[str, Any] | None = None,
        max_solve_iterations: int = 25,
    ):
        group_tuple = (groups,) if isinstance(groups, str) else tuple(groups)

        def decorator(func):
            cls._ensure_allowed_vars()
            arg_names = tuple(variables) if variables is not None else tuple(inspect.signature(func).parameters.keys())
            output_name = output or func.__name__
            all_vars = (output_name, *arg_names)
            solve_targets = solve_for or (output_name,)
            unknown = [v for v in all_vars if v not in cls.ALLOWED_VARS]
            if unknown:
                warnings.warn(
                    f"Relation variables {unknown!r} are not Reactor fields; they will not be auto-assigned",
                    UserWarning,
                )

            def equation(values, *, _func=func, _args=arg_names, _out=output_name):
                inputs = [values[arg] for arg in _args]
                return values[_out] - _func(*inputs)

            relation = Relation(
                name,
                all_vars,
                equation,
                priority=priority,
                rel_tol=rel_tol,
                solve_for=solve_targets,
                initial_guesses=initial_guesses,
                max_solve_iterations=max_solve_iterations,
            )
            cls._RELATIONS.append((group_tuple, relation))
            setattr(func, "relation", relation)
            return func

        return decorator

    @classmethod
    def get_relations(
        cls,
        groups: str | tuple[str, ...],
        *,
        require_all: bool = True,
        exclude: tuple[str, ...] | None = None,
    ) -> tuple[Relation, ...]:
        cls._ensure_relation_modules_loaded()
        requested = (groups,) if isinstance(groups, str) else tuple(groups)
        exclude_set = set(exclude or ())

        matches: list[Relation] = []
        seen: set[int] = set()
        for tags, rel in cls._RELATIONS:
            if exclude_set and exclude_set.intersection(tags):
                continue
            if require_all:
                if not all(tag in tags for tag in requested):
                    continue
            else:
                if not any(tag in tags for tag in requested):
                    continue
            if id(rel) in seen:
                continue
            seen.add(id(rel))
            matches.append(rel)
        return tuple(matches)


def relations_for(
    groups: str | tuple[str, ...],
    *,
    require_all: bool = True,
    exclude: tuple[str, ...] | None = None,
) -> tuple[Relation, ...]:
    return Reactor.get_relations(groups, require_all=require_all, exclude=exclude)


def confinement_relations(
    groups: str | tuple[str, ...] = ("confinement",),
    *,
    require_all: bool = True,
    exclude: tuple[str, ...] | None = None,
) -> tuple[Relation, ...]:
    return relations_for(groups, require_all=require_all, exclude=exclude)


def confinement_relations_by_name() -> dict[str, Relation]:
    return {rel.variables[0]: rel for rel in relations_for(("confinement",), require_all=False)}
