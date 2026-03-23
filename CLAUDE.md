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

## Results (200 steps): train_loss ~4.26, val_bpb ~2.49, 1319ms/step

## Next: Metal kernels (fused NS, RMSNorm+linear, relu²), TTT, CPU/GPU overlap
