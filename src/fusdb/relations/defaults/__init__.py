"""Default relation helpers.

Scalar/numeric defaults (geometry ``kappa``/``delta``/``squareness``, the
composition fractions ``f_D``/``f_T``/``f_He*``) now live as ``default`` metadata
in ``variables.yaml`` and are applied during compilation -- as held constants
when nothing can move them, or as balance-driven free cores when a constraint
(the steady-state particle balance, gated on ``tau_p``) determines them.  Only
the profile generators (a uniform profile from a scalar average) remain as
relations here, because they synthesise a profile rather than seed a scalar.
"""

from .profiles import *
