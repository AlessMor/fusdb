"""Coordinate-mapping variables layered onto the main variable registry.

The historical registry defined ``rho`` as normalized minor radius. The
profile-coordinate refactor makes ``rho`` framework state instead: a neutral
normalized computational sampling grid. Physical normalized coordinates and
integration measures are ordinary profile variables, so relations can declare
the geometry dependency explicitly without introducing another user-facing
class.

This small overlay keeps the large legacy ``variables.yaml`` stable while the
coordinate migration is staged. It updates the process-wide registry object in
place so existing importers of ``variable_registry.VARIABLES`` and newer
importers of ``registry.VARIABLES`` always share one canonical registry.
"""

from __future__ import annotations

from dataclasses import replace

from .variable_registry import VariableRegistry, VariableSpec


_RHO_LEGACY_ALIASES = {"normalized_minor_radius", "r_over_a"}

# These are physical mappings/integration measures, not independent profile
# unknowns. A supplied mapping is authoritative data; an unsupplied mapping must
# be produced deterministically by an active geometry relation.
PHYSICAL_COORDINATE_NAMES = frozenset({"rho_minor", "rho_tor", "v_norm", "w_V"})


def _coordinate_spec(
    name: str,
    description: str,
    *,
    aliases: tuple[str, ...] = (),
    domain: tuple[float | None, float | None, bool, bool] = (0.0, 1.0, True, True),
    default_relation: tuple[str, ...] = (),
) -> VariableSpec:
    return VariableSpec(
        name=name,
        aliases=aliases,
        unit="dimensionless",
        shape=1,
        domain=domain,
        solver_domain=domain,
        description=description,
        rel_tol=1.0e-3,
        abs_tol=1.0e-9,
        default_relation=default_relation,
    )


def with_coordinate_variables(base: VariableRegistry) -> VariableRegistry:
    """Apply the explicit profile-coordinate contract to ``base`` in place.

    Mutating the registry container is intentional here: registry metadata is
    process-global and immutable after package initialization, while several
    established modules import the singleton directly from
    ``variable_registry``. Keeping the object identity prevents a split-brain
    base/augmented registry during the staged migration.
    """
    if "rho_minor" in base and "w_V" in base:
        return base

    specs: list[VariableSpec] = []
    for spec in base:
        if spec.name == "rho":
            spec = replace(
                spec,
                aliases=tuple(alias for alias in spec.aliases if alias not in _RHO_LEGACY_ALIASES),
                description=(
                    "Common normalized computational profile coordinate, from axis/core centre "
                    "(rho=0) to the separatrix (rho=1). It is a sampling grid, not a physical "
                    "minor-radius or flux convention."
                ),
            )
        specs.append(spec)

    specs.extend(
        (
            _coordinate_spec(
                "rho_minor",
                "Normalized physical minor-radius mapping r/a tabulated on the common rho grid.",
                aliases=("normalized_minor_radius", "r_over_a", "minor_radius_coordinate"),
                default_relation=("Tokamak normalized minor-radius coordinate",),
            ),
            _coordinate_spec(
                "rho_tor",
                "Normalized toroidal-flux radius sqrt(Phi/Phi_edge) tabulated on the common rho grid.",
                aliases=("normalized_toroidal_flux_radius", "rho_toroidal"),
            ),
            _coordinate_spec(
                "v_norm",
                "Normalized enclosed plasma volume V(<rho)/V_p tabulated on the common rho grid.",
                aliases=("normalized_enclosed_volume", "enclosed_volume_fraction"),
                default_relation=("Tokamak normalized enclosed volume",),
            ),
            _coordinate_spec(
                "w_V",
                "Non-negative volume-integration weight proportional to dV/drho on the common rho grid.",
                aliases=("volume_integration_weight", "dV_drho_weight"),
                domain=(0.0, None, True, True),
                default_relation=("Tokamak volume integration weight",),
            ),
        )
    )
    augmented = VariableRegistry(
        specs,
        rel_tol_default=base.rel_tol_default,
        profile_size_default=base.profile_size_default,
    )
    base._specs = augmented._specs
    base._alias_to_name = augmented._alias_to_name
    return base
