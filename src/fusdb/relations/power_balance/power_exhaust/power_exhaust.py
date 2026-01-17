"""Power exhaust relations defined once."""

from fusdb.reactor_class import Reactor
from fusdb.relation_class import PRIORITY_STRICT
from fusdb.relation_util import nonzero, symbol


@Reactor.relation(
    "power_exhaust",
    name="P_sep ratio",
    output="P_sep_over_R",
    priority=PRIORITY_STRICT,
    constraints=(nonzero(symbol("R")),),
    initial_guesses={
        "P_sep_over_R": lambda v: v["P_sep"] / v["R"],
        "P_sep": lambda v: v["P_sep_over_R"] * v["R"],
        "R": lambda v: v["P_sep"] / v["P_sep_over_R"],
    },
)
def p_sep_ratio(P_sep: float, R: float) -> float:
    """Return the P_sep / R ratio."""
    return P_sep / R


@Reactor.relation(
    "power_exhaust",
    name="P_sep metric",
    output="P_sep_B_over_q95AR",
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
)
def p_sep_metric(P_sep: float, B0: float, q95: float, A: float, R: float) -> float:
    """Return the P_sep * B0 / (q95 * A * R) metric."""
    return P_sep * B0 / (q95 * A * R)


# TODO(high): add relations for P_fus_wall and P_n_wall
    # to do this, add a S_wall variable that defaults to S_p if not specified
    
# TODO(low): cfspopcon adds 
    # P_sep*B0/R0, which scales roughly the same as the parallel heat flux density entering the scrape-off-layer.
    # P_sep * B_pol / (R * n^2), which scales roughly the same as the impurity fraction required for detachment.