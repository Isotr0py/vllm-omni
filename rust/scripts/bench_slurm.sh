#!/usr/bin/env bash
# Submit the A/B frontend benchmark to a Slurm node.
# Usage: bash rust/scripts/bench_slurm.sh [model] [partition] [rates...]
# Results land in artifacts/vllm-omni-rs-bench-<date>/.
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Omni-3B}"
PARTITION="${2:-batch}"
RATES=("${@:3}")
[ "${#RATES[@]}" -eq 0 ] && RATES=(1 4 16 64 inf)

ROOT=/home/mozf/vllm-omni
OUTDIR=$ROOT/artifacts/vllm-omni-rs-bench-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTDIR"
echo "results dir: $OUTDIR"
# Nodes with orphaned VLLM::EngineCore procs squatting GPU memory (not ours).
EXCLUDE="${OMNI_RS_EXCLUDE:-gb200-rack1-02,gb200-rack1-03}"

srun -p "$PARTITION" -N1 --exclusive --exclude="$EXCLUDE" --job-name=omni-rs-bench \
  bash "$ROOT/rust/scripts/_bench_node.sh" "$MODEL" "$OUTDIR" "${RATES[@]}"
