search-and-learn/          ← you are here (cwd)
├── sal/
│   ├── config.py          # All config parameters
│   ├── models/
│   │   ├── reward_models.py   # PRM via Transformers
│   │   └── nli_similarity.py  # NLI similarity model
│   ├── search/
│   │   ├── dora.py            # DORA search (Transformers backend)
│   │   └── utils.py           # Beam, generate_step (replaces vLLM)
│   └── utils/
│       ├── data.py            # Dataset loading
│       ├── parser.py          # YAML + CLI arg parser
│       └── score.py           # Score aggregation
├── scripts/
│   ├── test_time_compute.py       # Main entry point
│   ├── run_dora_nli_qwen.sh       # NLI mode
│   ├── run_dora_hybrid_qwen.sh    # Hybrid mode
│   ├── run_dora_cosine_qwen.sh    # Cosine baseline
│   ├── run_ablations.sh           # All 7 ablation studies
│   └── run_full_evaluation.sh     # 3 models × 3 benchmarks
├── recipes/
│   └── Llama-3.2-1B-Instruct/
│       ├── dora_nli_qwen.yaml
│       └── dora_hybrid_qwen.yaml
├── allmodels/                     # Downloaded model weights
└── requirements.txt