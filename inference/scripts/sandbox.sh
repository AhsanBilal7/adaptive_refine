CONDA_BASE=$(conda info --base 2>/dev/null)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate sandbox-runtime

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SANDBOX_DIR="${SCRIPT_DIR}/../../SandboxFusion"

cd "${SANDBOX_DIR}"

make run-online PORT=9010 &
make run-online PORT=9020 &
make run-online PORT=9030 &
make run-online PORT=9040 &
make run-online PORT=9050 &
make run-online PORT=9060 &
make run-online PORT=9070 &
make run-online PORT=9080 &
