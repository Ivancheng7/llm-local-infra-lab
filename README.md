# llm-local-infra-lab

Rust CLI that analyzes local LLM metadata (config + safetensors index) and
produces precision-aware memory plans — no weights download, no GPU required.

Phase 1 target: **Qwen/Qwen3.8-27B** (`Qwen3_5ForConditionalGeneration`,
64 hybrid-attention layers = 48 linear + 16 full, vision tower, 1 MTP layer).

## Commands

```bash
# Inspect a HF config.json
cargo run -- inspect-config --path metadata/Qwen3.8-27B/config.json

# Inspect a safetensors index (shards, tensors, total bytes)
cargo run -- inspect-index --path metadata/Qwen3.8-27B/model.safetensors.index.json

# Full memory plan (markdown or json)
cargo run -- plan --metadata metadata/Qwen3.8-27B --precision bf16,int8,int4 --format markdown
cargo run -- plan --metadata metadata/Qwen3.8-27B --format json
```

## Fetching metadata (no 52 GB download)

```bash
mkdir -p metadata/Qwen3.8-27B
curl -L -o metadata/Qwen3.8-27B/config.json \
  https://hf-mirror.com/Qwen/Qwen3.8-27B/resolve/main/config.json
curl -L -o metadata/Qwen3.8-27B/model.safetensors.index.json \
  https://hf-mirror.com/Qwen/Qwen3.8-27B/resolve/main/model.safetensors.index.json
```

## Qwen3.8-27B verified numbers (from the official index)

- 1199 tensors across 18 shards
- Declared total: 55,562,855,904 bytes (≈51.75 GiB BF16)
- Implied parameter count: ~27.8B
- Layer split: 48 linear attention + 16 full attention (interval 4)
- KV cache (BF16, 16 full layers): 2,048 bytes/token/layer
  → 0.25 GiB @ 4K ctx, 16 GiB @ 262K ctx

## Evidence boundary

- Precision sizes (INT8/INT4) are estimates scaled from the checkpoint's
  declared BF16 `total_size`; the safetensors index carries no per-tensor dtype.
- KV estimates cover full-attention layers only; linear-attention layers hold
  fixed-size recurrent state, not per-token KV.
- Tensor classification is by name (module counts), not per-tensor byte counts.

## Roadmap

- Phase 1 (current): metadata planner — config/index parsing, tensor
  classification, precision estimates, KV estimates, markdown/JSON reports.
- Phase 2: local benchmark harness (TTFT/ITL/tok/s, VRAM, RSS, disk I/O)
  on rented hourly GPU time.
- Phase 3: MoE expert-cache/offload analysis.

## Development

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
```
