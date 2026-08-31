#!/usr/bin/env bash
# Node-side A/B benchmark: Python omni frontend vs vllm-omni-rs (Rust frontend).
# Invoked by bench_slurm.sh inside a Slurm allocation; not meant to be run
# directly. Args: <model> <outdir> [rates...]
set -uo pipefail

MODEL="${1:?model required}"
OUTDIR="${2:?outdir required}"
shift 2
RATES=("$@")
[ "${#RATES[@]}" -eq 0 ] && RATES=(1 4 16 64 inf)

ROOT=/home/mozf/vllm-omni
PY=$ROOT/.venv/bin/python
VLLM=$ROOT/.venv/bin/vllm
RS_BIN=$ROOT/rust/target/release/vllm-omni-rs
PORT=8091
INPUT_LEN=512
OUTPUT_LEN=256

export HF_HOME=${HF_HOME:-/mnt/lustre/hf-models}
mkdir -p "$OUTDIR"

# The Rust frontend cannot read composite omni configs; give it a text-only
# shim directory (tokenizer symlinks + hoisted config.json).
SHIM=$ROOT/artifacts/frontend-shims/$(basename "$MODEL")
if [ ! -e "$SHIM/config.json" ]; then
  "$PY" "$ROOT/rust/scripts/make_frontend_model_shim.py" "$MODEL" "$SHIM"
fi

cleanup() {
  # This job has the node exclusively; sweep orphaned engine procs.
  # Bracket patterns avoid matching this script's own cmdline.
  pkill -u "$(whoami)" -f '[t]arget/release/vllm-omni-rs' 2>/dev/null
  pkill -u "$(whoami)" -f '[r]ust_frontend' 2>/dev/null
  pkill -u "$(whoami)" -f '[b]in/vllm serve' 2>/dev/null
  pkill -u "$(whoami)" -f '[V]LLM::EngineCore' 2>/dev/null
  pkill -u "$(whoami)" -f '[S]tageEngineCoreProc' 2>/dev/null
  sleep 3
  true
}
trap cleanup EXIT

wait_health() {  # <logfile> <pid>
  for _ in $(seq 1 900); do
    curl -sf -m 2 localhost:$PORT/health >/dev/null 2>&1 && return 0
    kill -0 "$2" 2>/dev/null || break
    sleep 2
  done
  echo "server failed to become healthy, see $1" >&2
  tail -30 "$1" >&2
  return 1
}

tree_pids() {  # <pid> -> pid + all descendants
  local pid=$1
  echo "$pid"
  for c in $(pgrep -P "$pid" 2>/dev/null); do tree_pids "$c"; done
}

cpu_seconds() {  # <pid> -> user+sys seconds of the whole process tree
  local total=0 v
  for p in $(tree_pids "$1"); do
    v=$(awk '{print ($14+$15)/100}' "/proc/$p/stat" 2>/dev/null || echo 0)
    total=$(awk "BEGIN{print $total+$v}")
  done
  echo "$total"
}

run_bench() {  # <label> <frontend_pid>
  local label=$1 fe_pid=$2
  for rate in "${RATES[@]}"; do
    local n
    if [ "$rate" = inf ]; then n=256; else n=$((rate * 10)); fi
    local cpu0
    cpu0=$(cpu_seconds "$fe_pid")
    "$VLLM" bench serve \
      --backend openai-chat --endpoint /v1/chat/completions \
      --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
      --dataset-name random --random-input-len $INPUT_LEN \
      --random-output-len $OUTPUT_LEN --random-range-ratio 0.2 \
      --request-rate "$rate" --num-prompts "$n" \
      --extra-body '{"modalities":["text"]}' \
      --ignore-eos --seed 42 \
      --save-result --result-dir "$OUTDIR" \
      --result-filename "${label}_rate${rate}.json" \
      > "$OUTDIR/${label}_rate${rate}.log" 2>&1
    local rc=$?
    local cpu1
    cpu1=$(cpu_seconds "$fe_pid")
    echo "$label rate=$rate rc=$rc frontend_cpu_s=$(awk "BEGIN{print $cpu1-$cpu0}")" \
      >> "$OUTDIR/summary.txt"
  done
}

# ---- A: Python frontend (stock omni api server) ----
echo "=== starting python frontend ==="
"$VLLM" serve "$MODEL" --omni --port $PORT \
  > "$OUTDIR/python_server.log" 2>&1 &
A_PID=$!
wait_health "$OUTDIR/python_server.log" "$A_PID" || exit 1
# warmup (text-only: omni chat defaults to audio output, and the talker
# stage is broken in this environment — torch.cat empty-list crash)
curl -s -m 300 -X POST localhost:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"warmup\"}], \"modalities\": [\"text\"], \"max_tokens\": 32}" >/dev/null
run_bench python "$A_PID"
kill "$A_PID" 2>/dev/null
wait "$A_PID" 2>/dev/null
# sweep stage-proc orphans before the Rust run reclaims the GPUs
cleanup
sleep 10

# ---- B: Rust frontend (vllm-omni-rs) ----
echo "=== starting rust frontend ==="
VLLM_OMNI_RS_PYTHON="$PY" "$RS_BIN" serve --model "$SHIM" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port $PORT --ready-timeout-secs 1800 \
  --python "$PY" -- "$MODEL" \
  > "$OUTDIR/rust_server.log" 2>&1 &
B_PID=$!
wait_health "$OUTDIR/rust_server.log" "$B_PID" || exit 1
curl -s -m 300 -X POST localhost:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"warmup\"}], \"max_tokens\": 32}" >/dev/null
run_bench rust "$B_PID"
kill "$B_PID" 2>/dev/null
wait "$B_PID" 2>/dev/null

echo "=== done, results in $OUTDIR ==="
