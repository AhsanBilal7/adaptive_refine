# DN-BoN: Direction-Normalized Best-of-N

Implementation of **Greedy**, **Best-of-N (BoN)**, and **Direction-Normalized Best-of-N (DN-BoN)** for evaluating LLM math reasoning, based on the NeurIPS 2026 research proposal.

---

## Project Structure

```
dn_bon/
├── run_eval.py            ← Main entry point (CLI)
├── inference_methods.py   ← Greedy / BoN / DN-BoN implementations
├── model_wrapper.py       ← HuggingFace LM wrapper (no vLLM)
├── verifier.py            ← Outcome / Length / PRM verifiers
├── clustering.py          ← Semantic trace clustering for DN-BoN
├── evaluator.py           ← Evaluation loop + results I/O
├── dataset_loader.py      ← Unified dataset loading
├── parser.py              ← Answer extraction (from repo)
├── grader.py              ← Math equivalence checking (from repo)
├── utils.py               ← Small helpers for parser.py
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU quantisation (optional):
```bash
pip install bitsandbytes>=0.43.0
```

---

## Supported Datasets

| `--dataset`     | Source                                    |
|-----------------|-------------------------------------------|
| `math500`       | lighteval/MATH (500 test problems)        |
| `aime24`        | Maxwell-Jia/AIME_2024                     |
| `aime25`        | Maxwell-Jia/AIME_2025                     |
| `amc23`         | Maxwell-Jia/AMC_2023                      |
| `gsm8k`         | openai/gsm8k                              |
| `olympiadbench` | lmms-lab/OlympiadBench                    |
| `college_math`  | TIGER-Lab/college-math                    |
| `minerva_math`  | math-ai/minerva_math                      |
| `mmlu_stem`     | cais/mmlu (STEM categories)               |
| `custom`        | Local JSON/JSONL (use `--custom_path`)    |

---

## Usage

### 1. Greedy only

```bash
python run_eval.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset aime24 \
  --methods greedy \
  --max_samples 30
```

### 2. All three methods (N=8, MATH-500)

```bash
python run_eval.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset math500 \
  --methods greedy bon dn_bon \
  --N 8 \
  --max_samples 500 \
  --output_dir ./results/math500
```

### 3. Alpha ablation for DN-BoN

```bash
python run_eval.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset gsm8k \
  --alpha_ablation 0 0.5 1 \
  --N 16 \
  --max_samples 200
```
`alpha=0` exactly recovers plain BoN, making this a clean ablation.

### 4. Budget scaling (vary N)

```bash
for N in 4 8 16 32; do
  python run_eval.py \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --dataset math500 \
    --methods bon dn_bon \
    --N $N \
    --max_samples 100 \
    --output_dir ./results/scaling
done
```

### 5. Custom dataset

```bash
python run_eval.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset custom \
  --custom_path ./my_problems.jsonl \
  --methods greedy bon dn_bon
```

Custom JSONL format (one JSON object per line):
```json
{"question": "What is 2+2?", "answer": "4"}
{"question": "Solve x^2=9", "answer": "3"}
```

### 6. 4-bit quantised model (low VRAM)

```bash
python run_eval.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --load_in_4bit \
  --dataset aime24 \
  --methods greedy bon dn_bon \
  --N 8
```

### 7. PRM verifier (Process Reward Model)

```bash
python run_eval.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset math500 \
  --methods bon dn_bon \
  --verifier prm \
  --prm_model Qwen/Qwen2.5-Math-PRM-7B \
  --N 8
```

---

## Key Arguments

| Argument               | Default                    | Description                                      |
|------------------------|----------------------------|--------------------------------------------------|
| `--model`              | *(required)*               | HF model id or local path                        |
| `--dataset`            | *(required)*               | Benchmark name                                   |
| `--methods`            | `greedy bon dn_bon`        | Which methods to run                             |
| `--N`                  | `8`                        | Sample budget for BoN / DN-BoN                   |
| `--alpha`              | `0.5`                      | Normalisation exponent (0 = plain BoN)           |
| `--alpha_ablation`     | None                       | Run DN-BoN with multiple α values                |
| `--verifier`           | `outcome`                  | `outcome` / `length` / `prm`                     |
| `--embedding_model`    | `all-MiniLM-L6-v2`         | ST model for trace clustering                    |
| `--similarity_threshold` | `0.85`                   | Cosine threshold for grouping directions         |
| `--clustering_method`  | `agglomerative`            | `agglomerative` or `greedy`                      |
| `--cluster_on_answer`  | False                      | Cluster only the `\boxed{}` answer               |
| `--max_new_tokens`     | `1024`                     | Max generation length                            |
| `--temperature`        | `0.7`                      | Sampling temperature                             |
| `--top_p`              | `0.95`                     | Nucleus sampling p                               |
| `--max_samples`        | None                       | Limit number of evaluation problems              |
| `--output_dir`         | `./results`                | Directory for JSONL + summary JSON files         |
| `--hf_cache_dir`       | None                       | HuggingFace local cache dir                      |
| `--load_in_4bit`       | False                      | 4-bit NF4 quantisation (bitsandbytes)            |
| `--load_in_8bit`       | False                      | 8-bit quantisation (bitsandbytes)                |

---

## Output Files

For each `(method, dataset, N)` combination the following files are written to `--output_dir`:

```
results/
├── greedy_math500_N8.jsonl          ← per-question results
├── greedy_math500_N8_summary.json   ← accuracy + stats
├── bon_math500_N8.jsonl
├── bon_math500_N8_summary.json
├── dn_bon_math500_N8.jsonl
├── dn_bon_math500_N8_summary.json
└── comparison_math500_N8.txt        ← side-by-side comparison table
```

### Summary JSON structure

```json
{
  "accuracy": 0.6240,
  "n_correct": 312,
  "n_total": 500,
  "avg_elapsed_sec": 4.21,
  "redundancy_stats": {              // DN-BoN only
    "avg_num_clusters": 3.8,
    "avg_selected_cluster_size": 1.9,
    "avg_max_cluster_size": 4.2,
    "frac_selected_from_top_cluster": 0.31
  },
  "results": [ ... ]
}
```

---

## DN-BoN Method Summary

Standard BoN selects:

```
x* = argmax_i  V(x_i)
```

DN-BoN normalises by cluster size:

```
x* = argmax_i  V(x_i) / |C(x_i)|^α
```

| α value | Behaviour                              |
|---------|----------------------------------------|
| `0`     | Identical to standard BoN              |
| `0.5`   | **Default**: moderate direction penalty|
| `1`     | Full proportional penalty              |

---

## Recommended Models

| Model                               | VRAM  | Notes                        |
|-------------------------------------|-------|------------------------------|
| `Qwen/Qwen2.5-Math-7B-Instruct`     | ~16GB | Strong math, chat-templated  |
| `Qwen/Qwen2.5-Math-1.5B-Instruct`   | ~4GB  | Lightweight option           |
| `deepseek-ai/deepseek-math-7b-instruct` | ~16GB | Strong on competition math |
| `meta-llama/Llama-3.1-8B-Instruct`  | ~16GB | General purpose              |
| Any + `--load_in_4bit`              | ~6GB  | 4-bit quantised              |
