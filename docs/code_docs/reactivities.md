---

status: Online

---

# Reactivities

This page combines implementation notes, selected `mkdocstrings` entries, and
citations into the shared `fusdb` bibliography for reactivity relations.

## Overview

The reactivity layer provides source-specific `<sigma v>` relations for the
main fusion channels and a standalone interactive plotter for comparing those
sources. Narrative background lives in the
[cross-sections and reactivities knowledge page](../knowledgebase/plasma_physics/cross_sections_reactivities.md);
this page stays focused on code-facing entrypoints.

## Source References

- Bosch-Hale fits: [@bosch_hale_1992]
- Hively fits: [@hively_1977]
- CF88 parametrizations: [@caughlan_fowler_1988]
- NRL tabulated rates: [@beresnyak_2023_nrl_plasma_formulary]
- ENDF/B-VIII.0 and VIII.1 data: [@brown_2018_endfb_viii0; @nobre_2024_endfb_viii1]

## Selected Interfaces

### `sigmav_DT_Hively`

::: fusdb.relations.reactivities.reactivity_functions.sigmav_DT_Hively

### `sigmav_DT_ENDFB_VIII1`

::: fusdb.relations.reactivities.reactivity_functions.sigmav_DT_ENDFB_VIII1

### `sigmav_DD_BoschHale`

::: fusdb.relations.reactivities.reactivity_functions.sigmav_DD_BoschHale

### `sigmav_THe3_D_ENDFB_VIII1`

::: fusdb.relations.reactivities.reactivity_functions.sigmav_THe3_D_ENDFB_VIII1

### `reactivity_plotter`

::: fusdb.plotting.reactivity_plotter.reactivity_plotter

## See Also

- [Cross Sections and Reactivities](../knowledgebase/plasma_physics/cross_sections_reactivities.md)
- [Interactive Reactivity Plotter](reactivity_plotter.md)
- [Relation Modules API](api/relations.md)
- [Shared Bibliography](../bibliography.md)

## References

\bibliography
