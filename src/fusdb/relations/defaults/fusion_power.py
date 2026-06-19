"""Default fusion-power relation helpers."""

from fusdb import relation


@relation(
    name="Default zero advanced fusion powers",
    tags=("default", "fusion_power"),
    outputs=(
        "P_fus_He3He3",
        "P_fus_He3He3_alpha",
        "P_fus_He3He3_p",
        "P_fus_THe3",
        "P_fus_THe3_D",
        "P_fus_THe3_D_alpha",
        "P_fus_THe3_D_D",
        "P_fus_THe3_np",
        "P_fus_THe3_np_alpha",
        "P_fus_THe3_np_n",
        "P_fus_THe3_np_p",
        "P_fus_TT_alpha",
        "P_fus_TT_n",
    ),
)
def default_zero_advanced_fusion_powers() -> tuple[float, ...]:
    """Fallback zero power for omitted non-primary fusion branches.

    Returns:
        Zero power for each advanced fusion-power output.
    """
    return (0.0,) * 13
