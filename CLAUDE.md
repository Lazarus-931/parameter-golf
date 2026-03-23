# CLAUDE.md — var6-clean-int6

## Branch Purpose
Primary development branch for parameter-golf competition. Clean focused techniques, best val_bpb.

## Competition: 16 MB max, 10 min training (H100), score = val_bpb (lower better). Top: ~1.12 bpb

## Architecture: GPT + U-Net Skips
- 10 layers, dim=512, 8 heads, 4 KV heads (GQA), SwiGLU MLP (hidden=1088, mlp_mult=3.19)
- U-Net: 5 encoder + 5 decoder with learned skip_weights
- Tied embeddings (vocab=1024), RoPE (10000), logit softcap (30.0)
- Per-layer resid_mix blends current state with initial embedding
- SmearGate: per-dim gate blending token with previous (~512 params)
- BigramHash(4096): token-pair hash table (dim=128->512, ~524K params)
- muP: residual scales = 1/sqrt(2*num_layers) = 0.2236
- Ortho init (SVD) for large matrices, zero-init output projections
- No XSA (XSA_LAST_N=0). ~25.7M params, est ~15.5 MB compressed

## Quantization
- Per-row int6 (clip=31) all weights. fp16 embeddings + last c_k. fp32 control tensors. zstd-22

## Optimizer
- Muon: NS ortho (5 iter, pure MLX), WD=0.02, momentum 0.92->0.99/1500 steps, LR=0.02
- AdamW embed LR=0.03 WD=0.01 | AdamW scalar LR=0.02 WD=0.01
- Warmup 20, warmdown 3000 cosine, grad clip 0.3
- SWA: every 200 steps, last 50% warmdown

## Performance
- Uses `mx.fast.rms_norm` (fused kernel, ~19% faster than manual, supports autodiff)
- No custom Metal kernels (no VJP support breaks training autodiff)
- Newton-Schulz uses pure MLX ops (Metal kernel was slower than MLX matmul)

## Eval
- `EVAL_STRIDE=N` for overlapping evaluation windows (free val_bpb improvement)
- Stride=0 (default): standard non-overlapping eval
- Stride=512: ~2x eval cost, each scored token gets 512 extra context tokens

## Workflow
- Minis derek/lexie, secrets DEREK_SSH/LEXIE_SSH. SKIP_VAL=1 (~7 min). Timeout 360m
- NEVER run training on local Mac (crashes). Only small benchmarks locally.

## Results (200 steps, single Mac Mini, pre-SwiGLU)
- train_loss: 4.2627, val_bpb: 2.4944, step_avg: 1319ms, tok/s: 6161
- serialized: 15,051,865 bytes (zstd), SWA: 1 checkpoint (barely kicked in at 200 steps)

## Key Findings (Metal Kernels)
- Custom Metal kernels (`mx.fast.metal_kernel`) don't support VJP → can't use in training
- `@mx.custom_function` wrapper is 2x slower due to broken graph fusion
- Only viable for eval-only paths (not worth the complexity)
- `mx.fast.rms_norm` is the right approach: MLX built-in fused kernels that support autodiff

## Next
- Full 10-min training run to get real val_bpb with SwiGLU + hidden=1088
- Test `EVAL_STRIDE=512` for val_bpb improvement
- TTT (Test-Time Training) on validation set — only on remote machines
- GPTQ-lite post-training quantization refinement
