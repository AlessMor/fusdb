# TODO(high): from cfspopcon add Mavrin Coronal, Mavrin Noncoronal and PostJensen
# TODO(low): add impurity radiated power using the radas atomic data.
#
# TODO(cfspopcon, atomic-data dependent): port the core-seeded radiation chain. These
# need the Radas atomic-data tables (mean charge state Z(n_e, T_e) and L_z cooling
# curves) which fusdb does not yet have, so they are left as pointers rather than
# undecorated stubs:
#   - calc_intrinsic_radiated_power_from_core
#     (cfspopcon/formulas/radiated_power/intrinsic_radiated_power_from_core.py)
#   - calc_impurity_radiated_power (.../impurity_radiated_power/radiated_power.py)
#   - calc_impurity_charge_state (cfspopcon/formulas/impurities/impurity_charge_state.py)
#   - calc_core_radiator_conc / calc_edge_impurity_concentration
#     (cfspopcon/formulas/impurities/{core_radiator_conc.py,edge_radiator_conc.py})