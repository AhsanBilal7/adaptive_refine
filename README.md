# 📡Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling

Official implementation of **CRISP**, an inference-time reasoning algorithm for large language models (LLMs) proposed in our paper:

> **Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling**
> *ICLR 2026 paper*

**Paper:** https://arxiv.org/abs/2503.05188



## 🔥 Overview

We find that inference-time reward-model (RM) search methods such as BoN and MCTS often suffer from:

- RM hurts easy questions
- RM degrades with more samples
- High diversity breaks RM ranking

We propose **CRISP**, which:

- Clusters reasoning paths by final answers
- Aggregates reward at cluster level
- Iteratively updates prefixes
- Controls search diversity

CRISP improves reasoning accuracy by up to **5% over RM-based baselines** and **~10% over strong reasoning models**. 



## 🏗 Repository Structure

```
CRISP/
│
├── crisp_reason.py        # CRISP main algorithm
├── llm_reason.py          # main entry for experiments
├── beam_search.py         # beam search baseline
├── mcts_for_reasoning.py  # MCTS reasoning
├── mcts_backbone.py
├── prompts/
├── utils/
└── requirements.txt
```



## 🛠 Installation

```
git clone https://github.com/BugMakerzzz/CRISP.git
cd CRISP
pip install -r requirements.txt
```

Our tested environment: Python 3.10.0, CUDA 12.8.



## 🚀 Quick Start

Run CRISP on GSM8K:

```
python llm_reason.py \
  --method crisp \
  --model Qwen2_5_3b_chat \
  --dataset gsm8k \
  --reward skywork
```



# 📂 Data Path

All datasets are stored in the `data/` directory in **JSONL format**.

```
data/
├── csqa/
├── folio/
├── gpqa/
└── gsm8k/
    ├── train.jsonl
    └── dev.jsonl
...
```



# 🧩 Arguments

This section documents all CLI arguments for `llm_reason.py`.

### Core Settings

| argument        | default         | description           |
| --------------- | --------------- | --------------------- |
| `--model`       | Qwen2_5_3b_chat | policy model          |
| `--dataset`     | gsm8k           | dataset               |
| `--split`       | dev             | dev/test              |
| `--n_examples`  | 3               | few-shot              |
| `--seed`        | 17              | random seed           |
| `--remote`      | flag            | use API model         |
| `--test`        | flag            | debug mode (1 sample) |
| `--method`      | cot             | reasoning method      |
| `--n_samples`   | 5               | samples per question  |
| `--temperature` | 0.7             | sampling temp         |

Supported methods:

```
cot
sc
bestn
reward_sc
mcts
beam
crisp
```

------

### Reward Model

| argument   | default | description        |
| ---------- | ------- | ------------------ |
| `--reward` | None    | reward model       |
| `--agg`    | last    | reward aggregation |
| `--select` | reward  | selection strategy |

------

### MCTS Settings

| argument                    | default |
| --------------------------- | ------- |
| `--roll_num`                | 16      |
| `--max_depth_allowed`       | 3       |
| `--max_depth_allowed`       | 3       |
| `--num_votes`               | 10      |
| `--mcts_discount_factor`    | 1.0     |
| `--mcts_exploration_weight` | 1.0     |
| `--mcts_weight_scheduler`   | const   |
| `--mcts_num_last_votes`     | 1       |
| `--num_a1_steps`            | 5       |

------

### Beam Settings

| argument              | default |
| --------------------- | ------- |
| `--roll_num`          | 16      |
| `--beam_width`        | 2       |
| `--max_depth_allowed` | 3       |


------

### CRISP Settings

| argument              | default |
| --------------------- | ------- |
| `--roll_num`          | 16      |
| `--beam_width`        | 2       |
| `--max_depth_allowed` | 3       |
| `--ablation`          | None    |
| `--max_cls`           | 1       |



# 📄 Citation

```
@misc{li2026fixingbrokencompassdiagnosing,
      title={Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling}, 
      author={Jiachun Li and Pengfei Cao and Zhuoran Jin and Yubo Chen and Jiexin Xu and Huaijun Li and Xiaojian Jiang and Kang Liu and Jun Zhao},
      year={2026},
      eprint={2503.05188},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.05188}, 
}
```