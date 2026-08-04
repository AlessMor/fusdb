"""fusdb-reactors: the fusdb reactor scenario data, as an optional install.

The scenario YAML folders live next to this file.  The base ``fusdb``
package deliberately ships without them (S14: three install layers -- core
solvers+relations, reactor data, online-only docs); install this
distribution (``pip install ./reactors`` from a checkout) only when the
packaged scenarios are wanted.  ``Reactor.from_name`` resolves scenarios
through this package first, then a repository checkout.
"""

from pathlib import Path


def reactors_dir() -> Path:
    """Directory containing the packaged reactor scenario folders."""
    return Path(__file__).resolve().parent
