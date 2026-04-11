from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for test-time compute scaling experiments (Transformers backend)."""

    # ----- Model Paths -----
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    prm_path: str = "Qwen/Qwen2.5-Math-PRM-7B"
    embedding_path: Optional[str] = "BAAI/bge-m3"
    custom_chat_template: Optional[str] = None
    system_prompt: Optional[str] = None

    # ----- NLI Model Configuration -----
    nli_model_path: Optional[str] = None
    similarity_mode: str = "cosine"  # "cosine", "nli", or "hybrid"
    nli_lambda_penalty: float = 1.0
    nli_top_k: int = 10
    nli_batch_size: int = 32
    nli_max_length: int = 512
    nli_use_summary: bool = False
    nli_summary_max_steps: int = 5

    # ----- Dataset -----
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: Optional[str] = None
    dataset_split: str = "test"
    num_samples: Optional[int] = None

    # ----- Search Parameters -----
    approach: str = "dora"
    n: int = 64
    num_iterations: int = 40
    lookahead: int = 0
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 2048
    max_tokens_per_step: int = 256
    seed: int = 0

    # ----- REBASE / DORA Temperature -----
    rebase_temperature: float = 0.1
    balance_alpha: float = 0.01

    # ----- Allocation Strategy -----
    alpha_strategy: Optional[str] = None
    agg_strategy: str = "last"
    sort_completed: bool = True
    filter_duplicates: bool = False

    # ----- Device / Precision -----
    device: str = "auto"          # "auto", "cuda", "cpu"
    dtype: str = "bfloat16"       # "float16", "bfloat16", "float32"
    load_in_4bit: bool = False    # BitsAndBytes 4-bit quantization
    load_in_8bit: bool = False    # BitsAndBytes 8-bit quantization

    # ----- Batching -----
    search_batch_size: int = 1

    # ----- Output -----
    output_dir: str = "./results"
    output_file: Optional[str] = None

    # ----- Ablation Flags -----
    ablation_similarity: Optional[str] = None
    ablation_kmeans_k: int = 5
    ablation_hierarchical_threshold: float = 0.9