# CLAUDE.md -- var5-swiglu-unet

## Branch Purpose
This is the `var5-swiglu-unet` branch of a parameter-golf competition repo. The key change on this branch is replacing the relu-squared MLP with a **SwiGLU MLP** (`SiLU(x @ W_gate) * (x @ W_up)` then project down). It also enables XSA on the last 4 layers, uses a larger bigram vocab (10240), and tunes weight decay (0.04) and SWA parameters.

## Competition Rules
- **16 MB** max serialized model size (quantized + compressed)
- **10 minutes** max training wall-clock (evaluated on H100; we iterate on Mac Mini M4)
- Score = **val_bpb** (bits per byte) on held-out FineWeb validation set; lower is better
- Leaderboard top is around 1.12 bpb

## Architecture

### Model: GPT with U-Net Skip Connections
- **10 layers** (NUM_LAYERS=10), model_dim=512, 8 heads, 4 KV heads (GQA), head_dim=64
- **U-Net skips**: first 5 layers = encoder (accumulate skip tensors), last 5 = decoder (consume reversed skips via learned per-dim `skip_weights`, init 1.0)
- **Tied embeddings**: vocab=1024, tok_emb weight shared as LM head
- **RoPE** (base=10000), **logit softcap** (30.0)
- **Residual mixing**: per-layer learned `resid_mix` (2xD) blends current hidden state `x` with initial post-embedding state `x0`
- Per-block learnable `attn_scale` and `mlp_scale` (per-dim) on residual additions

### SwiGLU MLP (key change on this branch)
The MLP class uses SwiGLU activation instead of relu-squared:
- Three weight matrices: `w_gate` (dim -> hidden), `w_up` (dim -> hidden), `proj` (hidden -> dim)
- Forward: `proj(silu(w_gate(x)) * w_up(x))`
- Hidden dimension: `int(dim * mlp_mult * 2/3)` rounded up to nearest 64
- With dim=512 and mlp_mult=3: hidden = int(512 * 3 * 2/3) = 1024, already aligned to 64
- The 2/3 factor keeps total MLP params roughly comparable to 2-matrix relu-squared at the same mlp_mult

### Attention
- CausalSelfAttention with GQA (8 heads, 4 KV heads, group size 2)
- Q/K/V via CastedLinear (weights stored fp32, cast to bfloat16 on forward)
- RMSNorm applied to Q and K before RoPE application
- Learnable per-head `q_gain` (init 1.5) scales queries after RoPE
- `mx.fast.scaled_dot_product_attention` with causal mask
- **XSA (Exclusive Self-Attention)** on last 4 layers: subtracts the self-value projection from attention output, preventing tokens from attending to themselves. GQA-aware implementation.
- Output projection zero-initialized

### Special Features
- **SmearGate**: per-dim learnable gate (sigmoid, init 0) blending each token with previous token's embedding
- **BigramHashEmbedding**: hashes consecutive token pairs into 10240-bucket table (dim=128, projected to 512 via CastedLinear, scaled by learnable scalar init 0.05). Both embed and proj weights zero-initialized.

### Initialization
- **Orthogonal init** (SVD-based) for large weight matrices: c_q, c_k, c_v, w_gate, w_up
- **Zero init** for output projections: attn.proj, mlp.proj
- **muP scaling**: attn_scale and mlp_scale initialized to `1/sqrt(2*num_layers)` ~ 0.2236
- **Tied embedding**: Gaussian with std=0.005, stored as bfloat16

### Compute
- **bfloat16** compute dtype throughout
- **CastedLinear**: weights stored as float32, cast to compute dtype for matmul (`x @ W.astype(x.dtype).T`)

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| NUM_LAYERS | 10 | 5 encoder + 5 decoder |
| MODEL_DIM | 512 | |
| NUM_HEADS | 8 | |
| NUM_KV_HEADS | 4 | GQA, group size 2 |
| MLP_MULT | 3.0 | SwiGLU hidden = 512*3*2/3 = 1024 |
| XSA_LAST_N | 4 | XSA on layers 6-9 |
| VOCAB_SIZE | 1024 | SentencePiece BPE |
| TRAIN_SEQ_LEN | 1024 | |
| TRAIN_BATCH_TOKENS | 524288 | ~512 seqs per step |
| GRAD_ACCUM_STEPS | 8 | microbatch = 65536 tokens |
| ITERATIONS | 20000 | CI uses 200 for fast iteration |
| MAX_WALLCLOCK_SECONDS | 600 | 10 minutes |
| WARMDOWN_ITERS | 3000 | wallclock-based linear decay |
| BIGRAM_VOCAB_SIZE | 10240 | |
| BIGRAM_DIM | 128 | projected to model_dim=512 |
| WEIGHT_DECAY | 0.04 | Muon WD (higher than var6's 0.02) |
| GRAD_CLIP_NORM | 0.3 | |
| QK_GAIN_INIT | 1.5 | per-head learnable |
| LOGIT_SOFTCAP | 30.0 | |
| ROPE_BASE | 10000 | |
| TIED_EMBED_INIT_STD | 0.005 | |

## Quantization

Mixed int6 per-row quantization with zstd compression (`quantize_state_dict_mixed`):
- **Large 2D floating-point tensors** (MLP w_gate/w_up/proj, attention c_q/c_k/c_v/proj): **int6** per-row (clip_range=31, values in [-32, 31]). Scale stored as fp16 per row.
- **Tied embedding** (`tok_emb.weight`): kept as **fp16** (matches `FP16_KEEP_NAME_PATTERNS`)
- **Last-layer c_k** (`blocks.9.attn.c_k`): kept as **fp16** (special case)
- **Control tensors** (attn_scale, mlp_scale, resid_mix, q_gain, skip_weights, smear.gate, bigram.scale): kept in **fp32**, never quantized
- **Small tensors** (< 65536 elements): kept in fp16 or fp32 depending on pattern
- **Serialization**: pickle -> **zstd level 22** compression -> `.ptz` file
- After saving, a full roundtrip (decompress -> dequantize -> eval) reports the true post-quantization `val_bpb`

## Optimizer

Three-way split (`SplitOptimizers`):

1. **Muon** (2D matrix weights in blocks + bigram.proj):
   - SGD-momentum + Newton-Schulz orthogonalization (5 iterations)
   - LR: 0.02 (`matrix_lr`), WD: 0.04 (decoupled: `p *= 1 - lr * wd`)
   - Momentum: 0.99, warmup from 0.92 over 1500 steps
   - Nesterov-style: `g_eff = g + momentum * buf`
   - Scale factor: `sqrt(max(1, fan_out / fan_in))`

2. **AdamW** for embeddings (tok_emb.weight, bigram.embed.weight):
   - LR: 0.03, betas=(0.9, 0.95), eps=1e-8, WD=0.01

3. **AdamW** for scalars (skip_weights, smear.gate, bigram.scale, per-block scales/gains/resid_mix):
   - LR: 0.02, same betas/eps, WD=0.01

All LRs multiplied by `lr_mul` implementing wallclock-aware warmdown: when remaining wall-clock time < `warmdown_iters * avg_step_time`, LR linearly decays to 0.

### Stochastic Weight Averaging (SWA)
- Enabled by default
- Starts collecting when `lr_mul` drops below 0.6 (swa_start_frac=0.4)
- Snapshots every 50 steps during warmdown
- Final model uses arithmetic mean of all collected snapshots

## Workflow (CI)

File: `.github/workflows/train.yml`

- **Triggers**: push to any branch (if train script, data, cluster, or workflow files changed), or manual `workflow_dispatch`
- **Runner**: self-hosted Mac Mini with `parameter-golf` label
- **Host assignment**: branch name MD5 hash mod 2 picks either `derek` or `lexie`. SSH credentials in GitHub secrets (`DEREK_SSH`, `LEXIE_SSH`).
- **Concurrency**: one job per branch (`mini-<branch_name>`), no cancel-in-progress
- **Timeout**: 360 minutes
- **Data**: FineWeb 10B with SentencePiece 1024-token BPE, cached at `/opt/parameter-golf/data/`
- **CI defaults**: `SKIP_VAL=1`, `GRAD_ACCUM_STEPS=1`, `MLX_MAX_MICROBATCH_TOKENS=4096`, 200 iterations, 8192 batch tokens
- **Full training**: use env defaults (20k iters, 524k batch, 10 min cap)
- **Artifacts**: training log uploaded; results summarized in GitHub step summary

## Differences from Sibling Branches

| Feature | var5-swiglu-unet (this) | var6-clean-int6 | var2-xsa |
|---------|-------------------------|-----------------|----------|
| MLP | SwiGLU (3 matrices, hidden=1024) | relu-squared (2 matrices, hidden=1536) | relu-squared (2 matrices, hidden=1536) |
| XSA_LAST_N | 4 | 0 | 4 |
| WEIGHT_DECAY | 0.04 | 0.02 | 0.04 |
| BIGRAM_VOCAB_SIZE | 10240 | 4096 | 10240 |
| SWA_START_FRAC | 0.4 | 0.5 | 0.4 |
| SWA_EVERY | 50 | 200 | 50 |

## Key Files
- `train_gpt_mlx.py` -- single-file containing everything: model, optimizer, data loading, quantization, training loop. Must stay under 1500 lines.
- `.github/workflows/train.yml` -- CI workflow
- `data/cached_challenge_fineweb.py` -- dataset download/prep (called by CI if data not cached)

## Current Results
No final val_bpb results stored in repo. CI runs with SKIP_VAL=1 and 200 iterations for smoke testing. A full 10-minute run is needed for competition scoring. The key metric is `final_int8_zlib_roundtrip val_bpb` in the training log.

## What to Try Next
1. **Full-length training run** (10 min cap) to get real val_bpb and compare SwiGLU vs relu-squared
2. **Tune mlp_mult** -- try 4 (hidden = int(512*4*2/3) = 1365, rounded to 1408) for more capacity
3. **Tune XSA + SwiGLU interaction** -- XSA_LAST_N=4 may need different q_gain or scale init with SwiGLU
4. **Custom Metal kernels** -- fused Newton-Schulz (biggest throughput win), fused SwiGLU gate*up
5. **Test-Time Training (TTT)** -- score-first adaptation on val set
6. **GPTQ-lite** -- post-training quantization refinement to reduce int6 roundtrip error
7. **Reduce bigram vocab** if model size is tight (10240 * 128 = 1.3M params in bigram embedding alone)
8. **Try different SwiGLU hidden rounding** -- round to 128 or 256 for potentially better hardware utilization

## Environment
- Python 3.12, MLX (latest), numpy, sentencepiece, zstandard
- Mac Mini M4 (16GB unified memory) for iteration
- Final submission evaluated on H100
