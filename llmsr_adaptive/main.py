"""AdaptSci main entry point (standalone – no external llmsr dependency).

Backend options
───────────────
  Local Flask server (any HuggingFace model):
    python llm_engine/engine.py --model_path Qwen/Qwen2.5-7B-Instruct --port 5000 ...
    python main.py --local_server_url http://127.0.0.1:5000 ...

  OpenAI API:
    export API_KEY=sk-...
    python main.py --use_api True --api_model gpt-4o ...

  Anthropic Claude API:
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py --use_api True --api_model claude-3-5-sonnet-20241022 ...

  vLLM:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct --port 8000
    python main.py --llm_backend vllm --vllm_url http://127.0.0.1:8000 ...

Ablation modes
──────────────
  --ablation_mode full          → AdaptSci (adaptive buffer + Hessian)
  --ablation_mode buffer_only   → AdaptSci-Buffer ablation
  --ablation_mode hessian_only  → AdaptSci-Hessian ablation
  --ablation_mode none          → plain LLM-SR behaviour
"""
import os
import sys

# ── make 'adaptsci' importable when run as  python main.py  from this dir ────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

from adaptsci import pipeline
from adaptsci import config as adaptsci_config
from adaptsci.sampler import LocalLLM
from adaptsci.evaluator import LocalSandbox


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = ArgumentParser(description='AdaptSci: Adaptive Test-Time Compute for Equation Discovery')

# ── dataset / spec ────────────────────────────────────────────────────────────
parser.add_argument('--spec_path',    type=str, required=True,
                    help='Path to the problem specification file.')
parser.add_argument('--problem_name', type=str, default='oscillator1',
                    choices=['oscillator1', 'oscillator2', 'bactgrow', 'stressstrain'])
parser.add_argument('--log_path',     type=str, default='./logs/adaptsci_run',
                    help='Directory for TensorBoard logs and per-sample JSON files.')
parser.add_argument('--run_id',       type=int, default=1)

# ── LLM backend ───────────────────────────────────────────────────────────────
parser.add_argument('--llm_backend',      type=str,  default='local',
                    choices=['local', 'vllm'],
                    help='"local" = Flask engine.py server  |  "vllm" = vLLM server')
parser.add_argument('--local_server_url', type=str,  default='http://127.0.0.1:5000')
parser.add_argument('--vllm_url',         type=str,  default='http://127.0.0.1:8000')
parser.add_argument('--use_api',          type=bool, default=False)
parser.add_argument('--api_model',        type=str,  default='gpt-3.5-turbo',
                    help='gpt-3.5-turbo | gpt-4o | claude-3-5-sonnet-20241022 | …')

# ── AdaptSci ablation ─────────────────────────────────────────────────────────
parser.add_argument('--ablation_mode', type=str, default='full',
                    choices=['full', 'buffer_only', 'hessian_only', 'none'],
                    help=('"full"         → AdaptSci (buffer + Hessian)\n'
                          '"buffer_only"  → AdaptSci-Buffer ablation\n'
                          '"hessian_only" → AdaptSci-Hessian ablation\n'
                          '"none"         → plain LLM-SR (both disabled)'))

# ── Hessian config ────────────────────────────────────────────────────────────
parser.add_argument('--kappa_threshold',         type=float, default=50.0)
parser.add_argument('--coupling_threshold',      type=float, default=0.85)
parser.add_argument('--eff_dim_ratio_threshold', type=float, default=0.3)
parser.add_argument('--max_extra_restarts',      type=int,   default=3)
parser.add_argument('--hessian_step',            type=float, default=1e-4)

# ── Adaptive buffer config ────────────────────────────────────────────────────
parser.add_argument('--ab_window_size',        type=int,   default=50)
parser.add_argument('--ab_progress_threshold', type=float, default=0.01)
parser.add_argument('--ab_diversity_fn',       type=str,   default='token_jaccard',
                    choices=['token_jaccard', 'edit_distance', 'length_bucket'])

args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Build config
# ─────────────────────────────────────────────────────────────────────────────

cfg = adaptsci_config.Config(
    experience_buffer=adaptsci_config.ExperienceBufferConfig(),
    hessian=adaptsci_config.HessianConfig(
        enabled=True,
        condition_number_threshold=args.kappa_threshold,
        coupling_threshold=args.coupling_threshold,
        eff_dim_ratio_threshold=args.eff_dim_ratio_threshold,
        max_extra_restarts=args.max_extra_restarts,
        finite_diff_step=args.hessian_step,
    ),
    adaptive_buffer=adaptsci_config.AdaptiveBufferConfig(
        enabled=True,
        window_size=args.ab_window_size,
        progress_threshold=args.ab_progress_threshold,
        diversity_fn=args.ab_diversity_fn,
    ),
    use_api=args.use_api,
    api_model=args.api_model,
    ablation_mode=args.ablation_mode,
    llm_backend=args.llm_backend,
    local_server_url=args.local_server_url,
    vllm_url=args.vllm_url,
)

class_config = adaptsci_config.ClassConfig(
    llm_class=LocalLLM,
    sandbox_class=LocalSandbox,
)


# ─────────────────────────────────────────────────────────────────────────────
# Load spec and dataset
# ─────────────────────────────────────────────────────────────────────────────

with open(args.spec_path, encoding='utf-8') as f:
    specification = f.read()

df   = pd.read_csv(f'./data/{args.problem_name}/train.csv')
data = np.array(df)
X    = data[:, :-1]
y    = data[:, -1].reshape(-1)
if 'torch' in args.spec_path:
    X = torch.Tensor(X)
    y = torch.Tensor(y)
dataset = {'data': {'inputs': X, 'outputs': y}}


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    backend_str = (
        f'API ({args.api_model})' if args.use_api
        else f'{args.llm_backend} @ '
             f'{args.vllm_url if args.llm_backend == "vllm" else args.local_server_url}')

    print(f'\n{"="*66}')
    print(f'  AdaptSci')
    print(f'  problem      : {args.problem_name}')
    print(f'  ablation     : {args.ablation_mode}')
    print(f'  backend      : {backend_str}')
    print(f'  log_path     : {args.log_path}')
    print(f'{"="*66}\n')

    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=cfg,
        max_sample_nums=10000,
        class_config=class_config,
        log_dir=args.log_path,
    )
