- L-H switch for Popcon: popcon uses one regime for the whole map. _verify_all_regime_guards evaluates guards on system.complete(system.solver_values()) — a single nominal operating point. So a popcon that spans a wide (n, T) range picks one tau_E scaling at the nominal point and applies it to every grid cell, even in corners that are physically on the other side of the L-H boundary. Since a popcon exists precisely to explore operating space that crosses these thresholds, this is the one place the current design is genuinely under-serving the physics.

- match cfspocpon SPARC PRD

- match PROCESS

- rewrite tests