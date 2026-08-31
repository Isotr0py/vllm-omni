#!/usr/bin/env bash
# Smoke test: vllm-omni-rs serve with a real omni model on a Slurm node.
# Usage: bash rust/scripts/omni_smoke_slurm.sh [model] [partition]
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Omni-3B}"
PARTITION="${2:-batch}"
PORT=8091
ROOT=/home/mozf/vllm-omni
BIN=$ROOT/rust/target/release/vllm-omni-rs
PY=$ROOT/.venv/bin/python
# Nodes with orphaned VLLM::EngineCore procs squatting GPU memory (not ours).
EXCLUDE="${OMNI_RS_EXCLUDE:-gb200-rack1-02,gb200-rack1-03}"

# The Rust frontend cannot read composite omni configs; give it a text-only
# shim directory (tokenizer symlinks + hoisted config.json).
SHIM=$ROOT/artifacts/frontend-shims/$(basename "$MODEL")
if [ ! -e "$SHIM/config.json" ]; then
  HF_HOME=/mnt/lustre/hf-models "$PY" rust/scripts/make_frontend_model_shim.py "$MODEL" "$SHIM"
fi

srun -p "$PARTITION" -N1 --exclusive --exclude="$EXCLUDE" --job-name=omni-rs-smoke bash -c "
  set -euo pipefail
  export HF_HOME=/mnt/lustre/hf-models
  LOG=$ROOT/artifacts/omni-rs-smoke-\$(date +%H%M%S).log
  $BIN serve --model $SHIM --served-model-name '$MODEL' --host 127.0.0.1 \
    --port $PORT --ready-timeout-secs 1200 --python $PY -- '$MODEL' > \$LOG 2>&1 &
  SERVER_PID=\$!
  cleanup() {
    kill \$SERVER_PID 2>/dev/null
    sleep 2
    # This job has the node exclusively; sweep orphaned engine procs.
    # Bracket patterns avoid matching this script's own cmdline.
    pkill -u \$(whoami) -f '[t]arget/release/vllm-omni-rs' 2>/dev/null
    pkill -u \$(whoami) -f '[r]ust_frontend' 2>/dev/null
    pkill -u \$(whoami) -f '[V]LLM::EngineCore' 2>/dev/null
    pkill -u \$(whoami) -f '[S]tageEngineCoreProc' 2>/dev/null
    sleep 3
    true
  }
  trap cleanup EXIT
  for i in \$(seq 1 900); do
    curl -sf -m 2 localhost:$PORT/health >/dev/null 2>&1 && break
    kill -0 \$SERVER_PID 2>/dev/null || { echo '=== server died, log tail ==='; tail -50 \$LOG; exit 1; }
    sleep 2
  done
  curl -sf -m 2 localhost:$PORT/health >/dev/null || { tail -50 \$LOG; exit 1; }
  echo '=== server healthy, sending chat request ==='
  curl -sN -m 120 -X POST localhost:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"'$MODEL'\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"stream\": true, \"max_tokens\": 32}'
  echo
  echo '=== smoke OK ==='
  echo \"server log: \$LOG\"
"
