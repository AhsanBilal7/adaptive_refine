CONDA_BASE=$(conda info --base 2>/dev/null)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate srscientist
# MODEL_PATH="../models/gpt-oss-120b"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$(cd "$SCRIPT_DIR/../../models/gpt-oss-120b" && pwd)"

mkdir -p "$MODEL_PATH"

# For machines without internet access, set TIKTOKEN_RS_CACHE_DIR=CACHE_FILE by following the instructions in this issue: https://huggingface.co/openai/gpt-oss-120b/discussions/39.
python3 -m sglang.launch_server --model-path  $MODEL_PATH \
    --tp 8 \
    --tool-call-parser gpt-oss \
    --reasoning-parser gpt-oss \
    --mem-fraction-static 0.85


    # huggingface-cli download openai/gpt-oss-120b \
    #     --local-dir "$MODEL_PATH" \
    #     --local-dir-use-symlinks False