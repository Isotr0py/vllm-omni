# vllm-omni-rs (draft)

Rust HTTP frontend for vLLM-Omni, reusing the vLLM Rust frontend lib crates
(from a sibling vLLM checkout) and talking to vLLM-Omni's `AsyncOmni` through a
small Python façade process that speaks the vLLM v1 EngineCore ZMQ/msgpack
protocol.

```
HTTP client ──► vllm-omni-rs (Rust: OpenAI API, chat render, tokenize/detok)
                    │  ZMQ + msgpack (EngineCore v1, Bootstrapped mode)
                    ▼
              façade (python -m vllm_omni.entrypoints.rust_frontend)
                    │  in-process
                    ▼
              AsyncOmni ──► per-stage vLLM EngineCore procs (omni pipeline)
```

Phase-1 scope: text-only `/v1/chat/completions` + `/v1/completions` (+ the
other text endpoints the vLLM Rust server provides), single engine, no DP.
Audio/video/image routes are NOT served here — run the Python API server
(`vllm-omni serve --omni ...`) alongside for those.

## Build

Prerequisites:

- rustup with toolchain 1.95 (pinned by `rust-toolchain.toml`)
- the vLLM checkout at `../../vllm` (i.e. `/home/mozf/vllm`) — the crates are
  path dependencies; the checkout must stay put
- network access for git dependencies (llm-multimodal, oss-harmony)

```bash
cd rust
cargo build --release   # binary at target/release/vllm-omni-rs
```

## Run

Omni checkpoints use composite configs (`thinker_config.text_config...`) that
the Rust side cannot read — generate a text-only shim directory first:

```bash
export HF_HOME=/mnt/lustre/hf-models
.venv/bin/python rust/scripts/make_frontend_model_shim.py \
    Qwen/Qwen3-Omni-30B-A3B-Instruct artifacts/frontend-shims/Qwen3-Omni-30B-A3B-Instruct
```

`serve` spawns and supervises the Python façade itself (recommended):

```bash
./target/release/vllm-omni-rs serve \
    --model artifacts/frontend-shims/Qwen3-Omni-30B-A3B-Instruct \
    --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --host 0.0.0.0 --port 8091 \
    --python /home/mozf/vllm-omni/.venv/bin/python \
    -- Qwen/Qwen3-Omni-30B-A3B-Instruct   # omni engine args go after `--`
```

`--model` (the shim) is used by the Rust frontend itself (tokenizer, chat
template); `--served-model-name` is what clients send; the args after `--` go
to the omni engine (the real model id, plus any `vllm-omni serve` flags).

`frontend` connects to a façade you launched yourself:

```bash
python -m vllm_omni.entrypoints.rust_frontend \
    --input-address tcp://127.0.0.1:62101 \
    --output-address tcp://127.0.0.1:62102 \
    -- Qwen/Qwen3-Omni-30B-A3B-Instruct

./target/release/vllm-omni-rs frontend \
    --model artifacts/frontend-shims/Qwen3-Omni-30B-A3B-Instruct \
    --input-address tcp://127.0.0.1:62101 --output-address tcp://127.0.0.1:62102
```

The façade also has a GPU-less `--mock-echo` mode for protocol testing.

## Known environment gotchas (this cluster)

- Slurm nodes are requested whole (`srun -N1 --exclusive`); there is no GPU
  gres. GPU jobs must go through Slurm.
- gb200-rack1-02/-03 have orphaned `VLLM::EngineCore` processes squatting
  ~210 GiB of GPU memory (not from this project); the scripts exclude them via
  `OMNI_RS_EXCLUDE`.
- Qwen2.5-Omni is currently unusable even via the plain Python path: its
  transformers audio tower requires `flash_attn`, which is not installed.

## Scripts

- `scripts/make_frontend_model_shim.py` — build the text-only shim model dir
  the Rust frontend needs for composite omni checkpoints.
- `scripts/omni_smoke_slurm.sh [model] [partition]` — Slurm smoke test with a
  real omni model.
- `scripts/bench_slurm.sh [model] [partition] [rates...]` — Slurm A/B
  benchmark: Python frontend vs vllm-omni-rs.
- `scripts/render_results.py <bench-dir>` — render `RESULTS.md` from a bench
  output dir; latest results live in
  `artifacts/vllm-omni-rs-bench-*/RESULTS.md`.

## Limitations (Phase 1 draft)

- Text output only; omni multimodal endpoints stay on the Python API server.
- Single logical engine (engine_count=1); no DP, no HELLO handshake mode.
- `session_id` from the Rust protocol is dropped (no omni equivalent).
- LoRA, prompt_embeds, pooling endpoints unsupported (inherited from the Rust
  frontend itself).
