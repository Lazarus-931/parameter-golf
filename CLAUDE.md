# CLAUDE.md — var6-clean-int6

## Branch Purpose
Primary development branch for parameter-golf competition. Clean focused techniques, best val_bpb. Develop kernels and TTT here.

## Competition: 16 MB max, 10 min training (H100), score = val_bpb (lower better). Top: ~1.12 bpb

## Architecture: GPT + U-Net Skips
- 10 layers, dim=512, 8 heads, 4 KV heads (GQA), 3x MLP (hidden=1536), relu²
- U-Net: 5 encoder + 5 decoder with learned skip_weights
- Tied embeddings (vocab=1024), RoPE (10000), logit softcap (30.0)
- Per-layer resid_mix blends current state with initial embedding
- SmearGate: per-dim gate blending token with previous (~512 params)
- BigramHash(4096): token-pair hash table (dim=128->512, ~524K params)
- muP: residual scales = 1/sqrt(2*num_layers) = 0.2236
- Ortho init (SVD) for large matrices, zero-init output projections
- No XSA (XSA_LAST_N=0). ~24.7M params, 7.54 MB compressed

## Quantization
- Per-row int6 (clip=31) all weights. fp16 embeddings + last c_k. fp32 control tensors. zstd-22

## Optimizer
- Muon: NS ortho (5 iter), WD=0.02, momentum 0.92->0.99/1500 steps, LR=0.02
- AdamW embed LR=0.03 WD=0.01 | AdamW scalar LR=0.02 WD=0.01
- Warmup 20, warmdown 3000 cosine, grad clip 0.3
- SWA: every 200 steps, last 50% warmdown

## Workflow
- Minis derek/lexie, secrets DEREK_SSH/LEXIE_SSH. SKIP_VAL=1 (~7 min). Timeout 360m

## Results (200 steps, single Mac Mini)

### var6-clean-int6 (this branch)
- train_loss: 4.2627, val_bpb: 2.4944, step_avg: 1319ms, tok/s: 6161
- serialized: 15,051,865 bytes (zstd), SWA: 1 checkpoint (barely kicked in at 200 steps)

### var2-xsa (comparison baseline)
- train_loss: 4.0278, val_bpb: 3.1403, step_avg: 1319ms, tok/s: 6204
- serialized: 13,513,101 bytes (zstd), SWA: 4 checkpoints

**Key insight**: var6 has higher train_loss but much lower val_bpb (2.49 vs 3.14). The muP scaling + uniform int6 quantization produce weights that survive quantization much better. var2's mixed int5/int6 damages MLP weights more aggressively.

## Custom Metal Kernels
- **Fused NS B-matrix**: `B = b*A + c*A²` computed in one Metal dispatch instead of 3 MLX ops. 10% speedup on NS, ~1-2% total step time.

## Next: more Metal kernels (fused relu², fused RMSNorm), TTT, CPU/GPU overlap
