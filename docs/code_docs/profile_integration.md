# Profile Integration In Relations

This page defines the explicit-only profile integration convention.

## Convention

For any scalar quantity that is physically volume-integrated:

1. Write the relation function with the explicit integral in code using
   `fusdb.utils.integrate_profile_over_volume`.
2. Do not rely on `RelationSystem` to integrate profile outputs automatically.
3. Keep profile quantities explicit in the graph (for example
   `sigmav_DT_profile`), then pipe them into integrated relations.

## Fusion Example

`Rr_DT` should be expressed as:

```python
Rr_DT = integrate_profile_over_volume(f_D * f_T * n_i**2 * sigmav_DT_profile, V_p)
```

where `sigmav_DT_profile` is produced by its own relation from `T_i`.

## Thermal Pressure Example

`p_th` should be expressed as a volume average:

```python
p_th = KEV_TO_J * integrate_profile_over_volume(n_e*T_e + n_i*T_i, V_p) / V_p
```

## Bremsstrahlung Example

`P_brem` should use a local profile law integrated explicitly over volume.

## Notes On Jacobian

`integrate_profile_over_volume` uses the cfspopcon-style default
`d(V/V_p) = 2*rho drho` over `rho in [0, 1]`.

Volume consistency warnings (`V_p` vs geometry-derived volume) are still emitted
by `RelationSystem` once per solve/evaluate cycle when geometry inputs are available.
