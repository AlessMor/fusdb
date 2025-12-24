"""Power exhaust relations defined once."""

from fusiondb.relations_values import PRIORITY_STRICT, Relation
from fusiondb.relations_util import require_nonzero

PSEP_RELATIONS: tuple[Relation, ...] = (
    Relation(
        "P_sep ratio",
        ("P_sep_over_R", "P_sep", "R"),
        lambda v: (require_nonzero(v["R"], "R", "power relations") or v["P_sep_over_R"] - v["P_sep"] / v["R"]),
        priority=PRIORITY_STRICT,
        initial_guesses={
            "P_sep_over_R": lambda v: v["P_sep"] / v["R"],
            "P_sep": lambda v: v["P_sep_over_R"] * v["R"],
            "R": lambda v: v["P_sep"] / v["P_sep_over_R"],
        },
    ),
    Relation(
        "P_sep metric",
        ("P_sep_B_over_q95AR", "P_sep", "B0", "q95", "A", "R"),
        lambda v: (
            require_nonzero(v["q95"], "q95", "power relations")
            or require_nonzero(v["A"], "A", "power relations")
            or require_nonzero(v["R"], "R", "power relations")
            or v["P_sep_B_over_q95AR"] - v["P_sep"] * v["B0"] / (v["q95"] * v["A"] * v["R"])
        ),
        priority=PRIORITY_STRICT,
        initial_guesses={
            "P_sep_B_over_q95AR": lambda v: v["P_sep"] * v["B0"] / (v["q95"] * v["A"] * v["R"]),
            "P_sep": lambda v: v["P_sep_B_over_q95AR"] * v["q95"] * v["A"] * v["R"] / v["B0"],
            "B0": lambda v: v["P_sep_B_over_q95AR"] * v["q95"] * v["A"] * v["R"] / v["P_sep"],
            "q95": lambda v: v["P_sep"] * v["B0"] / (v["P_sep_B_over_q95AR"] * v["A"] * v["R"]),
            "A": lambda v: v["P_sep"] * v["B0"] / (v["P_sep_B_over_q95AR"] * v["q95"] * v["R"]),
            "R": lambda v: v["P_sep"] * v["B0"] / (v["P_sep_B_over_q95AR"] * v["q95"] * v["A"]),
        },
    ),
)
