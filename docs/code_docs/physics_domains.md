# Physics Domains

`fusdb` relations are grouped by domain under `src/fusdb/relations`.
Grouping keeps formulas coherent, supports selective solving by tags, and
makes cross-domain couplings explicit.

## Domain Map

| Domain | Typical outputs | Notes |
| --- | --- | --- |
| `geometry` | `V_p`, shape factors, cross-section quantities | Provides geometry primitives used by power and profile integrals. |
| `plasma_composition` | fractions, effective charge, species consistency | Couples fuel mix and impurity assumptions to many downstream equations. |
| `plasma_profiles` | peaking factors, profile scalings | Bridges averaged quantities with profile-resolved terms. |
| `plasma_pressure` | `p_th`, beta quantities | Couples density, temperature, magnetic field, and composition. |
| `plasma_current` | `I_p`, bootstrap/current drive terms | Links geometry, q limits, confinement, and heating/current drive. |
| `confinement` | `tau_E`, stored energy relationships | Connects global performance to power balance. |
| `power_balance` | fusion power, exhaust, radiation, auxiliaries | Main energy accounting loop; many implicit couplings live here. |
| `operational_limits` | density/beta/q constraints | Encodes feasible operating envelopes and warning/limit logic. |
| `separatrix` / `scrape_off_layer` | edge and exhaust boundary conditions | Couples core solution to boundary/engineering checks. |
| `impurities` | impurity-specific terms (when present) | Supplemental physics closures for composition/radiation. |

## Why This Matters

- Domain tags can be used in `solving_order` to control solve sequencing.
- Violations are easier to interpret when grouped by physical subsystem.
- New formulas should be added to the most physically coherent domain first,
  then linked across domains through explicit variable dependencies.
