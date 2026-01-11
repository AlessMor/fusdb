"""Power exhaust relations defined once."""

from fusdb.relation_class import PRIORITY_STRICT, Relation
from fusdb.relation_util import nonzero, symbol

PSEP_RELATIONS: tuple[Relation, ...] = (
    Relation(
        "P_sep ratio",
        ("P_sep_over_R", "P_sep", "R"),
        symbol("P_sep_over_R") - symbol("P_sep") / symbol("R"),
        priority=PRIORITY_STRICT,
        constraints=(nonzero(symbol("R")),),
        initial_guesses={
            "P_sep_over_R": lambda v: v["P_sep"] / v["R"],
            "P_sep": lambda v: v["P_sep_over_R"] * v["R"],
            "R": lambda v: v["P_sep"] / v["P_sep_over_R"],
        },
    ),
    Relation(
        "P_sep metric",
        ("P_sep_B_over_q95AR", "P_sep", "B0", "q95", "A", "R"),
        symbol("P_sep_B_over_q95AR")
        - symbol("P_sep") * symbol("B0") / (symbol("q95") * symbol("A") * symbol("R")),
        priority=PRIORITY_STRICT,
        constraints=(nonzero(symbol("q95")), nonzero(symbol("A")), nonzero(symbol("R"))),
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
