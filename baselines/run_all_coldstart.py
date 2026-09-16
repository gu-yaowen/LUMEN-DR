"""Batch runner for all cold-start experiments.

Runs the three external models (MRDDA, AdaDR, AutoDR) on all 3 datasets
(Bdataset, Cdataset, Fdataset) under both cold-start settings
(cold_drug, cold_disease) with 5 folds each.
"""
import subprocess
import sys
import os
import time

REPOS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# Model configurations: (model_name, script_path, working_dir, dataset_name_mapping)
MODELS = [
    {
        'name': 'MRDDA',
        'script': os.path.join(REPOS_DIR, 'MRDDA', 'cold_start_train.py'),
        'cwd': os.path.join(REPOS_DIR, 'MRDDA'),
        'dataset_map': {'Bdataset': 'Bdataset', 'Cdataset': 'Cdataset', 'Fdataset': 'Fdataset'},
    },
    {
        'name': 'AdaDR',
        'script': os.path.join(REPOS_DIR, 'AdaDR', 'AdaDR', 'cold_start_train.py'),
        'cwd': os.path.join(REPOS_DIR, 'AdaDR', 'AdaDR'),
        'dataset_map': {'Bdataset': 'Ldataset', 'Cdataset': 'Cdataset', 'Fdataset': 'Fdataset'},
    },
    {
        'name': 'AutoDR',
        'script': os.path.join(REPOS_DIR, 'AutoDR', 'code', 'cold_start_train.py'),
        'cwd': os.path.join(REPOS_DIR, 'AutoDR', 'code'),
        'dataset_map': {'Bdataset': 'Ldataset', 'Cdataset': 'Cdataset', 'Fdataset': 'Fdataset'},
    },
]

DATASETS = ['Bdataset', 'Cdataset', 'Fdataset']
SPLIT_MODES = ['cold_drug', 'cold_disease']


def run_experiment(model_config, dataset, split_mode):
    """Run a single cold-start experiment."""
    model_name = model_config['name']
    mapped_dataset = model_config['dataset_map'][dataset]

    cmd = [
        PYTHON,
        model_config['script'],
        '--dataset', mapped_dataset,
        '--split_mode', split_mode,
    ]

    print(f"\n{'='*70}")
    print(f"Running: {model_name} | {dataset} ({mapped_dataset}) | {split_mode}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CWD: {model_config['cwd']}")
    print(f"{'='*70}\n")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=model_config['cwd'],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - start_time

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n{model_name} | {dataset} | {split_mode}: {status} ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all cold-start experiments')
    parser.add_argument('--models', nargs='+', default=['MRDDA', 'AdaDR', 'AutoDR'],
                        help='Models to run')
    parser.add_argument('--datasets', nargs='+', default=['Bdataset', 'Cdataset', 'Fdataset'],
                        help='Datasets to run')
    parser.add_argument('--modes', nargs='+', default=['cold_drug', 'cold_disease'],
                        help='Split modes to run')
    args = parser.parse_args()

    total = 0
    success = 0
    failed = []

    for model_name in args.models:
        model_config = next(m for m in MODELS if m['name'] == model_name)
        for dataset in args.datasets:
            for split_mode in args.modes:
                total += 1
                ok = run_experiment(model_config, dataset, split_mode)
                if ok:
                    success += 1
                else:
                    failed.append(f"{model_name}|{dataset}|{split_mode}")

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY: {success}/{total} succeeded")
    if failed:
        print("Failed experiments:")
        for f in failed:
            print(f"  - {f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
