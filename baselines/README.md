# External baselines

This directory contains the patched source code used for the MRDDA, AdaDR, and
AutoDR comparisons in the LUMEN-DR study. The baseline implementations remain
separated because they use different graph libraries and dependency versions.

## Layout

```text
baselines/
├── MRDDA/                  # patched MRDDA source and B/C/F inputs
├── AdaDR/                  # patched AdaDR source and inputs
├── AutoDR/                 # patched AutoDR source and inputs
├── cold_start_splits/      # cold-start protocol documentation and split inputs
├── cold_start_utils.py     # shared cold-start helpers
├── run_all_coldstart.py    # optional sequential baseline runner
├── run_coldstart_batch.py  # optional parallel baseline runner
└── requirements.txt        # baseline environment record
```

The former `shared_splits/` directory and archived `formal_results_*.json`
files were intentionally removed. Therefore this checkout does not contain the
historical baseline metrics or the exact shared pair-level split archives.
Recreate the split files locally before running the cross-model scripts, and
record the split seed and checksum with the resulting experiment.

## Environment

The reported baseline runs used Python 3.10, PyTorch 2.2.0, DGL 1.1.2,
torch-geometric 2.7.0, torch-scatter 2.1.2, scikit-learn 1.7.2, NumPy 1.26.4,
pandas, SciPy, and NetworkX. Install the recorded dependencies with:

```bash
python -m pip install -r baselines/requirements.txt
```

Some baseline packages, especially DGL and torch-scatter, require a wheel that
matches the local PyTorch and CUDA/CPU build. Resolve that compatibility before
running a baseline.

## Individual entry points

Run each model from its own code directory, after supplying the expected local
dataset and split files:

```bash
cd baselines/MRDDA
python main.py -da Bdataset

cd ../AdaDR/AdaDR
python drug_train.py --data_name Ldataset --device -1

cd ../AutoDR/code
python main.py --dataset Fdataset --device cpu
```

`Ldataset` is the historical name used by AdaDR for the Bdataset inputs. The
cold-start protocol and the expected split conventions are documented in
[`cold_start_splits/cold_start_reproduction_protocol.md`](cold_start_splits/cold_start_reproduction_protocol.md).

## Compatibility patches

The checked-in baseline code contains only compatibility and evaluation fixes
needed for the reported comparisons, including modern DGL API updates, CPU-safe
device selection, validation-based model selection, and loading of externally
generated shared splits. The model-specific README files and source comments
retain the details of those changes.

## Scope and limitations

The baseline code is included for transparency and optional reruns. It is not
part of the installable `lumen-dr` package, and a fresh checkout cannot recreate
the archived paper tables without regenerating the omitted split and result
artifacts. The source code itself remains under the original baseline licenses.
