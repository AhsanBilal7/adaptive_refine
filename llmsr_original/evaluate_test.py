"""Evaluate the best discovered equation on ID and OOD test datasets.

Scans all  <log_path>/samples/*.json  files, picks the equation with the
highest training score, then runs the spec's evaluate() function on both
test_id.csv and test_ood.csv (parameters are re-optimised for each split,
which is the standard LLM-SR evaluation protocol).

Usage
─────
  # Auto-find best from a log directory:
python evaluate_test.py \
    --spec_path ./specs/specification_oscillator2_numpy.txt \
    --problem_name oscillator2 \
    --log_path ./logs/oscillator2_local

  # Evaluate a specific JSON sample directly:
python evaluate_test.py \
    --spec_path ./specs/specification_oscillator1_numpy.txt \
    --problem_name oscillator1 \
    --json_file ./logs/oscillator1_local/samples/samples_42.json
"""
import argparse
import glob
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

# ── make llmsr importable when run from this directory ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from llmsr import code_manipulation


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Score the best LLM-SR equation on ID and OOD test data.')
parser.add_argument('--spec_path',    type=str, required=True,
                    help='Spec .txt used during training.')
parser.add_argument('--problem_name', type=str, required=True,
                    choices=['oscillator1', 'oscillator2', 'bactgrow', 'stressstrain'])
parser.add_argument('--log_path',  type=str, default=None,
                    help='Log dir containing samples/*.json  (auto-finds best).')
parser.add_argument('--json_file', type=str, default=None,
                    help='Evaluate a specific JSON sample file directly.')
parser.add_argument('--data_dir',  type=str, default=None,
                    help='Override path to the data directory (default: ./data/<problem_name>).')
parser.add_argument('--timeout',   type=int, default=120,
                    help='Sandbox timeout per split in seconds (default 120).')
args = parser.parse_args()

if args.log_path is None and args.json_file is None:
    parser.error('Provide --log_path or --json_file.')


# ─────────────────────────────────────────────────────────────────────────────
# Find best equation from JSON logs
# ─────────────────────────────────────────────────────────────────────────────

def find_best_json(log_path: str) -> dict:
    """Scan all samples/*.json and return the entry with the highest score."""
    pattern = os.path.join(log_path, 'samples', '*.json')
    files   = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.join(log_path, '*.json'))
    if not files:
        raise FileNotFoundError(
            f'No sample JSON files found under: {log_path}\n'
            f'Make sure --log_path points to the directory used as log_dir '
            f'when running main.py.')

    best       = None
    best_score = -1e18
    for fpath in files:
        try:
            with open(fpath) as f:
                d = json.load(f)
            score = d.get('score')
            if score is None:
                continue
            score = float(score)
            if score > best_score:
                best_score    = score
                best          = d
                best['_file'] = fpath
        except Exception:
            continue

    if best is None:
        raise RuntimeError('No successfully scored samples found. All entries have null scores.')
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Build runnable program
# ─────────────────────────────────────────────────────────────────────────────

def build_program(spec: str, equation_fn_str: str, function_to_evolve: str) -> str:
    """Inject the best equation body into the spec template."""
    import copy
    template = code_manipulation.text_to_program(spec)
    eq_prog  = code_manipulation.text_to_program(equation_fn_str)
    eq_body  = eq_prog.get_function(function_to_evolve).body
    prog     = copy.deepcopy(template)
    prog.get_function(function_to_evolve).body = eq_body
    return str(prog)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess-isolated runner
# ─────────────────────────────────────────────────────────────────────────────

def _worker(program, function_to_run, dataset, q):
    try:
        ns = {}
        exec(program, ns)  # noqa: S102
        result = ns[function_to_run](dataset)
        q.put((float(result) if isinstance(result, (int, float)) else None, True))
    except Exception as e:
        print(f'  [eval error] {e}')
        q.put((None, False))


def run_evaluate(program: str, function_to_run: str,
                 dataset: dict, timeout: int) -> tuple:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker,
                                args=(program, function_to_run, dataset, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return None, False
    for _ in range(20):
        if not q.empty():
            return q.get_nowait()
        time.sleep(0.05)
    return None, False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── load spec ────────────────────────────────────────────────────────
    with open(args.spec_path, encoding='utf-8') as f:
        spec = f.read()

    function_to_run    = list(code_manipulation.yield_decorated(spec, 'evaluate', 'run'))[0]
    function_to_evolve = list(code_manipulation.yield_decorated(spec, 'equation', 'evolve'))[0]

    # ── get best equation ────────────────────────────────────────────────
    if args.json_file:
        with open(args.json_file) as f:
            entry = json.load(f)
    else:
        print(f'Scanning logs: {args.log_path}')
        entry = find_best_json(args.log_path)

    train_score     = entry.get('score')
    equation_fn_str = entry['function']
    sample_order    = entry.get('sample_order', '?')
    source_file     = entry.get('_file', args.json_file or '')

    print(f'\nBest sample  : #{sample_order}  ({os.path.basename(source_file)})')
    print(f'Train score  : {train_score}  (= −MSE on training data)')
    print(f'\nEquation:\n{"─"*56}')
    print(equation_fn_str.strip())
    print('─' * 56)

    # ── build full executable program ────────────────────────────────────
    program = build_program(spec, equation_fn_str, function_to_evolve)

    # ── load test datasets ────────────────────────────────────────────────
    data_root = args.data_dir or os.path.join(_HERE, 'data', args.problem_name)
    splits = {}
    for split in ('test_id', 'test_ood'):
        csv_path = os.path.join(data_root, f'{split}.csv')
        if not os.path.exists(csv_path):
            print(f'  [warning] {csv_path} not found — skipping {split}.')
            continue
        df  = pd.read_csv(csv_path)
        arr = np.array(df)
        splits[split] = {
            'inputs':  arr[:, :-1],
            'outputs': arr[:, -1].reshape(-1),
            'path':    csv_path,
        }

    if not splits:
        raise FileNotFoundError(
            f'No test CSVs found. Expected test_id.csv / test_ood.csv in {data_root}')

    # ── evaluate on each split ────────────────────────────────────────────
    print(f'\n{"="*56}')
    print(f'  LLM-SR Test Evaluation — {args.problem_name}')
    print(f'{"="*56}')

    results = {}
    for split_name, dataset in splits.items():
        score, ok = run_evaluate(
            program, function_to_run,
            {'inputs': dataset['inputs'], 'outputs': dataset['outputs']},
            args.timeout)
        if ok and score is not None:
            mse  = -score
            rmse = math.sqrt(max(mse, 0.0))
            nmse = mse / np.var(dataset['outputs'])   # NMSE = MSE / Var(y_true)
            results[split_name] = {'score': score, 'rmse': rmse, 'nmse': nmse}
            print(f'\n  {split_name.upper()}  ({dataset["path"]})')
            print(f'    Score (−MSE) : {score:.8f}')
            print(f'    RMSE         : {rmse:.8f}')
            print(f'    NMSE         : {nmse:.8f}  (MSE / Var(y))')
        else:
            results[split_name] = None
            print(f'\n  {split_name.upper()}')
            print(f'    FAILED  (timeout or runtime error)')

    # ── summary ───────────────────────────────────────────────────────────
    print(f'\n{"─"*56}')
    print(f'  Summary')
    print(f'{"─"*56}')
    print(f'  Train score : {train_score}')
    for split_name, res in results.items():
        if res:
            print(f'  {split_name:10s}: score={res["score"]:.8f}  '
                  f'RMSE={res["rmse"]:.8f}  NMSE={res["nmse"]:.8f}')
        else:
            print(f'  {split_name:10s}: FAILED')
    print(f'{"="*56}\n')


if __name__ == '__main__':
    main()
