# Scripts

## `data_prep/`

Scripts for sequence supplementation, CTD/STRING processing, protein-relation
construction, and final dataset normalization. They use the repository root as
their default location; external raw downloads and source datasets must be
provided separately when required.

## `experiments/`

The retained shell recipes correspond to the final training, ablation,
cold-start, and embedding-export runs. They calculate the repository root from
their own location and expose `src/` through `PYTHONPATH`, so they do not depend
on the original `/scratch` path. They still expect Linux/bash and a prepared
feature-cache directory.

## Quick start

For a simple standard run, use the portable wrapper from the repository root:

```bash
bash scripts/run_cv.sh Bdataset example_bdataset cuda:0
```

For detailed final-run settings, inspect the corresponding script under
`scripts/experiments/` and record the command configuration in the generated
output directory.
