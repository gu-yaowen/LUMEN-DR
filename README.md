# LUMEN-DR

LUMEN-DR is a multimodal heterogeneous-graph framework for drug--disease
association (DDA) prediction and computational drug repositioning. It combines
frozen pretrained representations of molecular structures, disease descriptions,
and protein sequences with relation-aware graph learning and an auxiliary
contrastive objective.

The repository contains the model implementation, final benchmark input files,
data-preparation code, baseline source code, and the documentation needed to
understand the reported experiments. Generated feature caches, training outputs,
paper figures, archived result summaries, and shared pair-level split files are
not distributed in this release.

## Repository layout

```text
.
├── src/lmdda/              # installable LUMEN-DR Python package
│   ├── models/             # HGT backbone and DDA prediction head
│   ├── losses/             # contrastive objective
│   └── utils/              # data, features, splits, metrics, and I/O
├── data/                   # final B/C/F dataset inputs and reference data
├── baselines/              # MRDDA, AdaDR, AutoDR source and notes
├── scripts/data_prep/      # dataset preparation and normalization scripts
├── scripts/run_cv.sh       # small standard-CV launcher
├── docs/                   # data and reproducibility documentation
├── pyproject.toml          # package metadata and dependencies
└── results/                # generated locally; ignored by Git
```

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch installation can depend on the available CPU/GPU and CUDA runtime. If
the default wheel is not suitable, install the matching PyTorch wheel first and
then run `python -m pip install -e . --no-deps` followed by the remaining
scientific Python dependencies. Baseline-specific dependencies are optional:

```bash
python -m pip install -e ".[baselines]"
```

Check the command-line interface:

```bash
python -m lmdda --help
# or, after installation:
lumen-dr --help
```

## Input data and feature caches

The final CSV inputs are stored under `data/Bdataset`, `data/Cdataset`, and
`data/Fdataset`. The model expects frozen feature caches under:

```text
features/<DATASET>/MolFormer_drug_emb.pkl
features/<DATASET>/BERT_disease_emb.pkl
features/<DATASET>/ESMC_protein_emb.pkl
```

The feature caches are deliberately excluded because of their size and because
they depend on the selected pretrained-model runtime. The current release does
not contain the original feature-generation entry point; therefore a complete
rerun requires regenerating these three cache files with the same encoder
versions, preprocessing rules, and entity ordering. See
[`docs/reproducibility.md`](docs/reproducibility.md) and
[`data/dataset_process.md`](data/dataset_process.md) before running training.

## Training

Run from the repository root after the feature caches are available:

```bash
python -m lmdda \
  --mode train \
  --dataset Bdataset \
  --run-name example_bdataset \
  --data-root data \
  --features-root features \
  --output-root outputs \
  --device cuda:0
```

For a quick command wrapper:

```bash
bash scripts/run_cv.sh Bdataset example_bdataset cuda:0
```

Use `--device cpu` on a CPU-only machine. The command writes checkpoints, logs,
fold metrics, predictions, and training curves to `outputs/<DATASET>/<RUN_NAME>/`.
These files are generated artifacts and are ignored by Git.

To score a candidate-pair CSV with a trained checkpoint:

```bash
python -m lmdda \
  --mode predict \
  --dataset Bdataset \
  --checkpoint outputs/Bdataset/example_bdataset/folds/fold_0/best.ckpt \
  --candidate-csv path/to/candidates.csv \
  --data-root data \
  --features-root features \
  --output-root outputs \
  --device cuda:0
```

## Reproducibility notes

The repository records the final normalized datasets and the data-processing
logic, but not every historical run artifact. The exact omissions and the
requirements for reconstructing an experiment are documented in
[`docs/reproducibility.md`](docs/reproducibility.md). Baseline source code is
retained under `baselines/`, while the archived shared split files and result
JSON summaries requested for removal are not included.

## Data preparation

The scripts under `scripts/data_prep/` document the sequence supplementation,
CTD/STRING processing, protein-relation preparation, and final index
normalization used for the released data. Some preparation steps require raw
CTD/STRING downloads or source data that are excluded by `.gitignore`; consult
[`data/dataset_process.md`](data/dataset_process.md) for provenance and
expected inputs.

## Baselines

The patched MRDDA, AdaDR, and AutoDR source code is retained for inspection and
optional reruns. Their original environments differ from the main LUMEN-DR
environment; see [`baselines/README.md`](baselines/README.md). The shared split
archives and archived result JSON files are intentionally absent, so exact
baseline reproduction requires reconstructing the documented splits locally.

## Citation and license

The repository currently contains third-party baseline implementations under
their respective licenses. Before public release, add the project citation and
an explicit repository-level license once the authors have agreed on them.
