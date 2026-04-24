# different template format of the commonsense example generation
CASE_FORMAT_LIST = ['..., then ...', '..., so ...', '..., and ...', '..., but ...', '..., after that, ...', 'Before ..., ...']

# openai config
OPENAI_API_KEY = ''
MAX_REQUESTS_PER_MINUTE = 3500 # 3_000 * 0.5
MAX_TOKENS_PER_MINUTE = 90000 #250_000 * 0.5
REQUEST_URL = ''

# ---------------------------------------------------------------------------
# All paths are HuggingFace Hub repo IDs.
# With HF_HOME=/scratch/$USER/hf_cache set, HF resolves from local cache
# automatically — no download occurs if the model is already cached.
# The {dir} placeholder has been removed from all paths.
# ---------------------------------------------------------------------------

# LLaMA
llama2_7b_path          = 'meta-llama/Llama-2-7b-hf'
llama2_7b_chat_path     = 'meta-llama/Llama-2-7b-chat-hf'
llama2_13b_path         = 'meta-llama/Llama-2-13b-hf'
llama2_13b_chat_path    = 'meta-llama/Llama-2-13b-chat-hf'
llama3_8b_path          = 'meta-llama/Meta-Llama-3-8B'
llama3_8b_chat_path     = 'meta-llama/Meta-Llama-3-8B-Instruct'
llama3_1_8b_chat_path   = 'meta-llama/Llama-3.1-8B-Instruct'
llama_moe_path          = 'llama-moe/LLaMA-MoE-v1-3_5B-4_16'

# Mistral
mistral_7b_path         = 'mistralai/Mistral-7B-v0.1'
mistral_7b_chat_path    = 'mistralai/Mistral-7B-Instruct-v0.2'

# Vicuna
vicuna_7b_path          = 'lmsys/vicuna-7b-v1.5'
vicuna_13b_path         = 'lmsys/vicuna-13b-v1.5'

# Qwen
qwen1_8b_path           = 'Qwen/Qwen-1_8B'
qwen2_5_3b_chat_path    = 'Qwen/Qwen2.5-3B-Instruct'
qwen2_5_7b_chat_path    = 'Qwen/Qwen2.5-7B-Instruct'
qwen2_5_14b_chat_path   = 'Qwen/Qwen2.5-14B-Instruct'

qwen2_5_math_1_5b_path      = 'Qwen/Qwen2.5-Math-1.5B'
qwen2_5_math_1_5b_chat_path = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
qwen2_5_math_7b_path        = 'Qwen/Qwen2.5-Math-7B'
qwen2_5_math_7b_chat_path   = 'Qwen/Qwen2.5-Math-7B-Instruct'

# Phi
phi2_path               = 'microsoft/phi-2'
phi3_path               = 'microsoft/Phi-3-small-8k-instruct'

# Yi
yi_1_5_6b_chat_path     = '01-ai/Yi-1.5-6B-Chat'

# Gemma
gemma_2_9b_path         = 'google/gemma-2-9b'
gemma_2_9b_chat_path    = 'google/gemma-2-9b-it'

# QwQ / DeepSeek
qwq_32b_path            = 'Qwen/QwQ-32B-Preview'
deepseek_14b_path       = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'

# ---------------------------------------------------------------------------
# Reward models
# ---------------------------------------------------------------------------
reward_skywork_path     = 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2'
reward_shepherd_path    = 'peiyi9979/math-shepherd-mistral-7b-prm'
reward_armorm_path      = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
reward_grm_path         = 'Ray2333/GRM-gemma2-2B-rewardmodel-ft'
reward_skyworko1_path   = 'Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B'
reward_genprm_path      = 'TIGER-Lab/GenPRM-7B'
reward_qwen_math_path   = 'Qwen/Qwen2.5-Math-PRM-7B'
reward_gemma_path       = 'UniqueGao/RM-Gemma-2B'  # verify this matches your cached version

figure_colors = ['#90BCD5', '#E76254', '#7976A2', '#4A5E65', '#E29957', '#86B5A1', '#B95A58', '#4292C6']