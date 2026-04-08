set -x
CONDA_BASE=$(conda info --base 2>/dev/null)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARQUET_FILE_PATH="$(cd "$SCRIPT_DIR/../../data/inference" && pwd)/llmsrbench.parquet"
OUTPUT_JSON_PATH="$(cd "$SCRIPT_DIR/.." && pwd)/output/memory_default/gpt-oss-20b-tool2_0_001_s40_max_25_top3_N1_test.json"
mkdir -p "$(dirname "$OUTPUT_JSON_PATH")"

# MODEL_PATH="../models/Qwen3-Coder-30B-A3B-Instruct"
MODEL_PATH="$(cd "$SCRIPT_DIR/../../models/gpt-oss-20b" && pwd)"


MODEL_URL="http://0.0.0.0:30000/v1"
SANDBOX_URLS=(
    "http://127.0.0.1:9010/run_code"
    "http://127.0.0.1:9020/run_code"
    "http://127.0.0.1:9030/run_code"
    "http://127.0.0.1:9040/run_code"
    "http://127.0.0.1:9050/run_code"
    "http://127.0.0.1:9060/run_code"
    "http://127.0.0.1:9070/run_code"
    "http://127.0.0.1:9080/run_code"
)


MAPE_THRESHOLD=0.001
NUM_TURNS=25
MAX_ASSISTANT_TURNS=25
TOP_K=3
# MAPE_THRESHOLD=0.001
# NUM_TURNS=5
# MAX_ASSISTANT_TURNS=5
# TOP_K=3
SOURCE=(
    "lsr_synth/bio_pop_growth"
    )


conda activate srscientist
cd inference/


TIMESTAMP=$(TZ='UTC-8' date +'%Y%m%d_%H%M%S')


BASENAME=$(basename "$OUTPUT_JSON_PATH" .json)

# 3. Get the output directory and replace "output" with "log"
OUTPUT_DIR=$(dirname "$OUTPUT_JSON_PATH")
LOG_DIR="${OUTPUT_DIR/output/log}"

# 4. Construct the full log file path
LOG_FILE="${LOG_DIR}/${BASENAME}_${TIMESTAMP}.log"

# 5. Create the log and output directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"



mkdir -p "$(dirname "$OUTPUT_JSON_PATH")"
echo "--- Starting main.py execution at $(TZ='UTC-8' date) ---" | tee -a "$LOG_FILE"

python main.py \
    --model-name "$MODEL_PATH" \
    --model-url "$MODEL_URL" \
    --sandbox-urls "${SANDBOX_URLS[@]}" \
    --parquet-file-path "$PARQUET_FILE_PATH" \
    --mape-threshold $MAPE_THRESHOLD \
    --num-turns $NUM_TURNS \
    --max-assistant-turns $MAX_ASSISTANT_TURNS \
    --top-k $TOP_K \
    --source "${SOURCE[@]}" \
    --output-json-path "$OUTPUT_JSON_PATH" 2>&1 | tee "$LOG_FILE"
    
echo "--- Finished main.py execution at $(TZ='UTC-8' date) ---" | tee -a "$LOG_FILE"
