from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, ClassVar

import warnings

from fusdb.geometry.plasma_geometry import (
    FRC_SHAPE_RELATIONS,
    GEOMETRY_RELATIONS,
    MIRROR_SHAPE_RELATIONS,
    SPHERICAL_TOKAMAK_SHAPE_RELATIONS,
    STELLARATOR_SHAPE_RELATIONS,
    TOKAMAK_SHAPE_RELATIONS,
)
from fusdb.plasma_parameters import PLASMA_RELATIONS
from fusdb.power_exhaust.power_exhaust import PSEP_RELATIONS
from fusdb.confinement.scalings import (
    CONFINEMENT_RELATIONS_BY_NAME,
    STELLARATOR_CONFINEMENT_RELATIONS,
    TOKAMAK_CONFINEMENT_RELATIONS,
)
from fusdb.relations_values import PRIORITY_RELATION, Relation, RelationSystem
from fusdb.relations_util import REL_TOL_DEFAULT


@dataclass
class Reactor:
    ALLOWED_REACTOR_CLASSES: ClassVar[tuple[str, ...]] = (
        "ARC-class", 
        "STEP-class", 
        "DEMO-class",
        )
    ALLOWED_REACTOR_CONFIGURATIONS: ClassVar[tuple[str, ...]] = (
        "tokamak",
        "compact tokamak",
        "spherical tokamak",
        "stellarator",
    )

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
        if self.allow_relation_overrides is None:
            self.allow_relation_overrides = False
        self.reactor_class = self._normalize_choice(
            self.reactor_class, self.ALLOWED_REACTOR_CLASSES, "reactor_class"
        )
        self.reactor_configuration = self._normalize_choice(
            self.reactor_configuration, self.ALLOWED_REACTOR_CONFIGURATIONS, "reactor_configuration"
        )
        rel_tol = REL_TOL_DEFAULT
        self._apply_geometry(rel_tol)
        self._apply_tokamak_shape(rel_tol)
        self._apply_plasma_parameters(rel_tol)
        self._apply_power_exhaust(rel_tol)
        # final source map cleanup
        self._sources = dict(self._sources)

    def _record_sources(self, var_map: dict[str, Any]) -> None:
        for name, var in var_map.items():
            if hasattr(var, "source"):
                self._sources[name] = var.source

    def _apply_geometry(self, rel_tol: float) -> None:
        lock = not bool(self.allow_relation_overrides)
        relations = [rel.with_tol(rel_tol) for rel in GEOMETRY_RELATIONS]
        config = (self.reactor_configuration or "").lower()
        if "spherical tokamak" in config:
            relations.extend(rel.with_tol(rel_tol) for rel in SPHERICAL_TOKAMAK_SHAPE_RELATIONS)
        system = RelationSystem(relations, rel_tol=rel_tol, warn=warnings.warn, lock_explicit=lock)
        for k, v in {
            "R": self.R,
            "a": self.a,
            "A": self.A,
            "R_max": self.R_max,
            "R_min": self.R_min,
            "Z_max": self.Z_max,
            "Z_min": self.Z_min,
            "kappa": self.kappa,
            "kappa_95": self.kappa_95,
            "delta": self.delta,
            "delta_95": self.delta_95,
        }.items():
            system.set(k, v)
        values = system.solve()

        self.R = values["R"]
        self.a = values["a"]
        self.A = values["A"]
        self.R_max = values["R_max"]
        self.R_min = values["R_min"]
        self.Z_max = values["Z_max"]
        self.Z_min = values["Z_min"]
        self.kappa = values["kappa"]
        self.kappa_95 = values["kappa_95"]
        self.delta = values["delta"]
        self.delta_95 = values["delta_95"]
        self._record_sources(system.var_map)

    def _apply_tokamak_shape(self, rel_tol: float) -> None:
        concept = (self.reactor_configuration or "").lower()
        lock = not bool(self.allow_relation_overrides)
        if "tokamak" not in concept:
            return

        system = RelationSystem(
            [rel.with_tol(rel_tol) for rel in TOKAMAK_SHAPE_RELATIONS], rel_tol=rel_tol, warn=warnings.warn, lock_explicit=lock
        )
        for k, v in {
            "R": self.R,
            "a": self.a,
            "kappa": self.kappa,
            "kappa_95": self.kappa_95,
            "delta_95": 0.0 if self.delta_95 is None else self.delta_95,
            "delta": self.delta,
            "squareness": 0.0 if self.squareness is None else self.squareness,
            "V_p": self.V_p,
            "S_p": self.S_p,
        }.items():
            system.set(k, v)
        values = system.solve()
        self.V_p = values["V_p"]
        self.S_p = values["S_p"]
        self.delta_95 = values["delta_95"]
        self.squareness = values["squareness"]
        self.kappa_95 = values.get("kappa_95")
        self.delta = values.get("delta")
        self._record_sources(system.var_map)

    def _apply_power_exhaust(self, rel_tol: float) -> None:
        lock = not bool(self.allow_relation_overrides)
        system = RelationSystem([rel.with_tol(rel_tol) for rel in PSEP_RELATIONS], rel_tol=rel_tol, warn=warnings.warn, lock_explicit=lock)
        for k, v in {
            "P_sep": self.P_sep,
            "P_sep_over_R": self.P_sep_over_R,
            "P_sep_B_over_q95AR": self.P_sep_B_over_q95AR,
            "R": self.R,
            "A": self.A,
            "B0": self.B0,
            "q95": self.q95,
        }.items():
            system.set(k, v)
        values = system.solve()
        self.P_sep = values["P_sep"]
        self.P_sep_over_R = values["P_sep_over_R"]
        self.P_sep_B_over_q95AR = values["P_sep_B_over_q95AR"]
        self._record_sources(system.var_map)

    def _apply_plasma_parameters(self, rel_tol: float) -> None:
        lock = not bool(self.allow_relation_overrides)
        relations = [rel.with_tol(rel_tol) for rel in PLASMA_RELATIONS]
        confinement_var = None

        # If tau_E is provided, choose the closest matching scaling (or validate the declared one).
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
            method_rel = CONFINEMENT_RELATIONS_BY_NAME.get(self.tau_E_method)
            if method_rel is None:
                raise ValueError(f"Unknown tau_E_method {self.tau_E_method!r}. Must match a confinement relation name.")
            relations.append(method_rel.with_tol(rel_tol))
            confinement_var = method_rel.variables[0]
            relations.append(
                Relation(
                    f"Energy confinement method {self.tau_E_method}",
                    ("tau_E", confinement_var),
                    lambda v: v["tau_E"] - v[confinement_var],
                    priority=PRIORITY_RELATION,
                ).with_tol(rel_tol)
            )

        system = RelationSystem(relations, rel_tol=rel_tol, warn=warnings.warn, lock_explicit=lock)
        values_to_set = {
            "n_e": self.n_e,
            "n_i": self.n_i,
            "T_e": self.T_e,
            "T_i": self.T_i,
            "p_th": self.p_th,
            "W_th": self.W_th,
            "V_p": self.V_p,
            "P_loss": self.P_loss,
            "tau_E": self.tau_E,
            "B0": self.B0,
            "B_p": self.B_p,
            "beta_T": self.beta_T,
            "beta_p": self.beta_p,
            "beta": self.beta,
        }
        if confinement_var:
            values_to_set[confinement_var] = self.tau_E

        for k, v in values_to_set.items():
            system.set(k, v)
        values = system.solve()
        self.n_e = values["n_e"]
        self.n_i = values["n_i"]
        self.T_e = values["T_e"]
        self.T_i = values["T_i"]
        self.p_th = values["p_th"]
        self.W_th = values["W_th"]
        self.P_loss = values["P_loss"]
        self.tau_E = values["tau_E"]
        if confinement_var:
            setattr(self, confinement_var, values[confinement_var])
        self.beta_T = values["beta_T"]
        self.beta_p = values["beta_p"]
        self.beta = values["beta"]
        self._record_sources(system.var_map)


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
