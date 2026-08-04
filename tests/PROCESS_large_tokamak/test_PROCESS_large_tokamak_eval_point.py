"""fusdb vs PROCESS at the large-tokamak evaluation design point.

Reference: ``reference/eval_point/`` -- PROCESS run on
``examples/data/large_tokamak_eval_IN.DAT`` with ``ioptimz = -2``, i.e. **no
optimisation**.  That matters: against the optimisation run fusdb would have to
be handed PROCESS's converged major radius, field, current and profiles, so much
of what came out would be downstream of what went in.  In evaluation mode the
input file *is* the design vector and everything else is forward-computed, so
every difference here is attributable to physics rather than to bookkeeping.

Every compared quantity agrees within 10%.  What it took, beyond the obvious
variable mapping, is recorded in ``README.md``; the four that mattered most:

* PROCESS's ``kappa`` is the SEPARATRIX elongation, not the areal one fusdb's
  ``kappa`` means, and its ``kappa_ipb`` is a third value again.
* PROCESS keeps heating power, transport loss power and separatrix power as
  three distinct quantities; fusdb's default collapses the first two.
* Mavrin's Lz already contains impurity bremsstrahlung, so fusdb's Z_eff-weighted
  ``P_brem`` double-counts it.
* fusdb's ``f_D``/``f_T``/``f_He4`` are fractions of TOTAL ion density, while
  PROCESS quotes helium against electrons and the D-T split against fuel ions.
"""

from __future__ import annotations

import pytest

from _process_compare import COMPARED, FIELDS, TOLERANCE, compare
from _process_fixture import EVAL_POINT


@pytest.fixture(scope="module")
def comparison():
    """Solve the eval-point fixture once for the whole module."""
    return compare(EVAL_POINT)


def test_reconcile_certifies(comparison):
    """The fixture solves cleanly -- no failed relations, no violated inputs."""
    _, result = comparison
    assert result["success"], f"failed relations: {result['failed_relations']}"
    assert not result["failed_relations"]
    assert not result["inputs_beyond_tolerance"]


def test_stays_in_h_mode(comparison):
    """fusdb agrees with PROCESS that this is an H-mode point.

    Not incidental: the regime is *derived*.  With the confinement enhancement
    correctly applied, less auxiliary power is needed, so ``P_sep`` falls toward
    the L-H threshold -- and an over-estimated threshold demotes the whole solve
    to ohmic, where an ohmic scaling wins the provider slot and returns a 39 s
    confinement time.  Selecting PROCESS's own L-H scaling
    (``i_l_h_threshold = 19``) is what keeps the two codes in the same regime.
    """
    _, result = comparison
    assert result["regime"] == "h_mode"


def test_all_fields_present(comparison):
    """Every field in the comparison surface actually resolved on both sides."""
    fields, _ = comparison
    missing = sorted(set(COMPARED) - set(fields))
    assert not missing, f"not computed by fusdb or absent from the MFILE: {missing}"


@pytest.mark.parametrize("name", sorted(COMPARED))
def test_within_tolerance(comparison, name):
    """Each compared quantity is within 10% of PROCESS."""
    fields, _ = comparison
    entry = fields[name]
    assert abs(entry["rel_error"]) <= TOLERANCE, (
        f"{name} ({entry['label']}): fusdb {entry['fusdb']:.6g} vs "
        f"PROCESS {entry['process']:.6g} = {entry['rel_error']:+.2%}"
    )


def test_volume_is_exact(comparison):
    """``V_p`` reproduces PROCESS exactly, not merely within tolerance.

    ``kappa_ipb`` is *defined* as V_p / (2 pi^2 R a^2), so supplying it and
    letting "IPB elongation from volume" run backwards recovers PROCESS's own
    volume.  fusdb's default Sauter form, fed the same elongation, runs -7.2%,
    which propagates into W_th, P_loss and P_aux -- so this is load-bearing
    rather than cosmetic.
    """
    fields, _ = comparison
    assert abs(fields["V_p"]["rel_error"]) < 1e-6
