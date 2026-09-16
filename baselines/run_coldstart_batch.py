"""Batch runner for all cold-start experiments.

Checks which experiments are already complete (by looking for formal_results
files) and runs only the missing ones.  Supports parallel execution with a
configurable number of concurrent workers.

Usage:
    python run_coldstart_batch.py                    # run all missing, 1 at a time
    python run_coldstart_batch.py --workers 2        # run 2 experiments in parallel
    python run_coldstart_batch.py --datasets Bdataset # only Bdataset
    python run_coldstart_batch.py --modes cold_drug   # only cold_drug
    python run_coldstart_batch.py --models MRDDA      # only MRDDA
    python run_coldstart_batch.py --dry_run           # show what would run
"""
import os
import sys
import time
import json
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

REPOS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPOS_DIR, 'cold_start_results')
PYTHON = sys.executable

# Model configurations
MODELS = {
    'MRDDA': {
        'script': os.path.join(REPOS_DIR, 'MRDDA', 'cold_start_train.py'),
        'cwd': os.path.join(REPOS_DIR, 'MRDDA'),
        'dataset_map': {'Bdataset': 'Bdataset', 'Cdataset': 'Cdataset', 'Fdataset': 'Fdataset'},
    },
    'AdaDR': {
        'script': os.path.join(REPOS_DIR, 'AdaDR', 'AdaDR', 'cold_start_train.py'),
        'cwd': os.path.join(REPOS_DIR, 'AdaDR', 'AdaDR'),
        'dataset_map': {'Bdataset': 'Ldataset', 'Cdataset': 'Cdataset', 'Fdataset': 'Fdataset'},
    },
    'AutoDR': {
        'script': os.path.join(REPOS_DIR, 'AutoDR', 'code', 'cold_start_train.py'),
        'cwd': os.path.join(REPOS_DIR, 'AutoDR', 'code'),
        'dataset_map': {'Bdataset': 'Ldataset', 'Cdataset': 'Cdataset', 'Fdataset': 'Fdataset'},
    },
}

DATASETS = ['Bdataset', 'Cdataset', 'Fdataset']
SPLIT_MODES = ['cold_drug', 'cold_disease']


def is_experiment_complete(model_name, dataset, split_mode):
    """Check if an experiment is already complete by looking for formal_results file."""
    results_file = os.path.join(
        RESULTS_DIR,
        f'formal_results_{model_name.lower()}_coldstart_{dataset}_{split_mode}.json')
    return os.path.exists(results_file)


def run_single_experiment(model_name, dataset, split_mode):
    """Run a single cold-start experiment."""
    config = MODELS[model_name]
    mapped_dataset = config['dataset_map'][dataset]

    cmd = [PYTHON, config['script'],
           '--dataset', mapped_dataset,
           '--split_mode', split_mode]

    log_file = os.path.join(RESULTS_DIR, f'log_{model_name}_{dataset}_{split_mode}.txt')

    print(f"\n{'='*70}")
    print(f"Running: {model_name} | {dataset} ({mapped_dataset}) | {split_mode}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CWD: {config['cwd']}")
    print(f"Log: {log_file}")
    print(f"{'='*70}\n")

    start_time = time.time()
    with open(log_file, 'w') as lf:
        result = subprocess.run(
            cmd, cwd=config['cwd'],
            stdout=lf, stderr=subprocess.STDOUT,
            text=True)
    elapsed = time.time() - start_time

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit={result.returncode})"
    print(f"\n{model_name} | {dataset} | {split_mode}: {status} ({elapsed:.1f}s)")

    return {
        'model': model_name,
        'dataset': dataset,
        'split_mode': split_mode,
        'success': result.returncode == 0,
        'elapsed': elapsed,
        'log_file': log_file,
    }


def main():
    parser = argparse.ArgumentParser(description='Batch runner for cold-start experiments')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        choices=DATASETS, help='Datasets to run')
    parser.add_argument('--modes', nargs='+', default=SPLIT_MODES,
                        choices=SPLIT_MODES, help='Split modes to run')
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                        choices=list(MODELS.keys()), help='Models to run')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would run without executing')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if results exist')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build list of experiments to run
    experiments = []
    skipped = []
    for model in args.models:
        for dataset in args.datasets:
            for mode in args.modes:
                if not args.force and is_experiment_complete(model, dataset, mode):
                    skipped.append((model, dataset, mode))
                else:
                    experiments.append((model, dataset, mode))

    print(f"\nCold-Start Batch Runner")
    print(f"  Models:   {args.models}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Modes:    {args.modes}")
    print(f"  Workers:  {args.workers}")
    print(f"  To run:   {len(experiments)}")
    print(f"  Skipped:  {len(skipped)} (already complete)")

    if skipped:
        print(f"\n  Skipped experiments:")
        for m, d, s in skipped:
            print(f"    {m} | {d} | {s}")

    if args.dry_run:
        print(f"\n  Experiments to run:")
        for m, d, s in experiments:
            print(f"    {m} | {d} | {s}")
        return

    if not experiments:
        print("\nAll experiments already complete!")
        return

    # Run experiments
    results = []
    start_all = time.time()

    if args.workers == 1:
        # Sequential execution
        for model, dataset, mode in experiments:
            result = run_single_experiment(model, dataset, mode)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_single_experiment, m, d, s): (m, d, s)
                for m, d, s in experiments
            }
            for future in as_completed(futures):
                m, d, s = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"\nERROR: {m} | {d} | {s}: {e}")
                    results.append({
                        'model': m, 'dataset': d, 'split_mode': s,
                        'success': False, 'elapsed': 0, 'error': str(e)
                    })

    total_elapsed = time.time() - start_all

    # Summary
    print(f"\n{'='*70}")
    print(f"Batch Summary ({total_elapsed:.1f}s total)")
    print(f"{'='*70}")
    successes = sum(1 for r in results if r['success'])
    failures = len(results) - successes
    for r in results:
        status = "OK" if r['success'] else "FAIL"
        print(f"  {status}  {r['model']:8s} | {r['dataset']:10s} | {r['split_mode']:15s}  ({r['elapsed']:.1f}s)")
    print(f"\n  Total: {len(results)}, Success: {successes}, Failed: {failures}")

    if failures > 0:
        print(f"\n  Failed experiments:")
        for r in results:
            if not r['success']:
                print(f"    {r['model']} | {r['dataset']} | {r['split_mode']}")
                if 'log_file' in r:
                    print(f"      Check log: {r['log_file']}")

    # Save batch summary
    summary_file = os.path.join(RESULTS_DIR, 'batch_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'total': len(results),
            'success': successes,
            'failed': failures,
            'total_elapsed': total_elapsed,
            'results': results,
        }, f, indent=2)
    print(f"\n  Summary saved to {summary_file}")


if __name__ == '__main__':
    main()
