# Variable

`Variable` stores a single quantity, its metadata, and value history across passes.

**Core fields**
- `name`, `unit`, `rel_tol`, `abs_tol`, `method`, `input_source`, `fixed`
- `values`: list of values (chronological)
- `value_passes`: pass id for each value
- `history`: list of change records

**Methods (concise, all)**
- `current_value` property returns most recent value.
- `input_value` property returns the original input/default value (cached).
- `get_value_at_pass(pass_id)` returns the value at or before a pass.
- `add_value(value, pass_id=None, reason=None, relation=None, default_rel_tol=0.0)` appends if changed.
- `get_from_dict(variables_dict, name, pass_id=None, allow_override=False, mode="current")` convenience getter.

**Example (Python)**
```python
from fusdb import Variable

var = Variable(
    name="R",
    values=[3.2],
    unit="m",
    rel_tol=0.02,
    input_source="explicit",
)
```
