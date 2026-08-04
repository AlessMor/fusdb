# PROCESS reference data

All 15 PROCESS runs the comparison is measured against, plus the scripts that
generate them. Files are gzipped (42 MB raw, 9.2 MB stored); `_process_mfile.py` reads
`.DAT` and `.DAT.gz` interchangeably, so paths can be written either way.

Vendored from [github.com/ukaea/PROCESS](https://github.com/ukaea/PROCESS)
@ `83d9f63fbd0d1085f343a1af964409942262ee6f`. Every run was produced with the
same PROCESS build (`3.4.2.dev101+g83d9f63fb`), which matters for the sweeps:
a differential comparison is only meaningful if the reference points share a
version.

| directory | runs | used by |
| --- | --- | --- |
| `eval_point/` | 1 | `test_..._eval_point.py` |
| `tungsten_sweep/w_*/` | 5 | `test_..._tungsten.py` |
| `plasma_variants/<tag>/` | 9 | `test_..._plasma_variants.py` |
| `introduction/` | 1 | not used by the tests — see below |
| `scripts/` | — | the `.ex.py` drivers that regenerate all of the above |

Each run directory holds its `IN.DAT` (the exact input used), `MFILE.DAT` (the
machine-readable key/value output) and `OUT.DAT`. Keep `OUT.DAT`: it carries the
switch *semantics* and the labelled power-balance table, and it is what settled
the `i_rad_loss = 1` question — that PROCESS subtracts only **core** radiation
from the power fed to the confinement scaling.

## Regenerating

The three `fusdb_*.ex.py` scripts in `scripts/` live in PROCESS's own
`examples/` directory upstream. Copy them there and run from the PROCESS root:

```bash
python examples/fusdb_eval_point.ex.py        # 1 run
python examples/fusdb_tungsten_sweep.ex.py    # 5 runs
python examples/fusdb_plasma_variants.ex.py   # 9 runs
```

They write into `examples/output/<name>/`, touching nothing existing. Check
`ifail = 1` on each before using a run as a reference.

`introduction.ex.py` and `single_model_evaluation.ex.py` are PROCESS's own
examples, kept here unmodified for reference.

## Why the evaluation runs, not the optimisation

All the runs the tests use are `ioptimz = -2` — evaluation mode, no optimiser.
Against an optimisation run fusdb must be handed PROCESS's converged major
radius, field, current, profiles and composition, so a large part of what comes
out is downstream of what went in. In evaluation mode the input file *is* the
design vector, so each difference is attributable to physics.

`introduction/` is a genuine run of the example's own input file (`ifail = 1`,
`sqsumsq = 2.35e-9`) and is kept only for `OPTIMUM.md`, which records how far the
optimum moves between that input and the `tests/` variant carried in the PROCESS
repository — a caution against seeding a fixture from the wrong one.
