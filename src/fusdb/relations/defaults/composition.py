"""Default composition relation helpers."""

from fusdb import relation


@relation(
    name="Default equimolar DT fuel fractions",
    tags=("default", "plasma", "composition", "dt"),
    outputs=("f_D", "f_T"),
)
def default_equimolar_dt_fuel_fractions() -> tuple[float, float]:
    """Fallback pure-DT fuel composition: f_D = f_T = 0.5.

    This is only appropriate when the scenario is explicitly a DT case and no
    supplied or non-default composition model provides f_D/f_T or n_D/n_T.
    """
    return 0.5, 0.5


@relation(
    name="Default no minority fuel fractions",
    tags=("default", "plasma", "composition", "dt"),
    outputs=("f_He3", "f_He4", "f_Imp"),
)
def default_no_minority_fuel_fractions() -> tuple[float, float, float]:
    """Fallback no-minority composition for simple pure-DT cases."""
    return 1.0e-10, 1.0e-10, 1.0e-10
