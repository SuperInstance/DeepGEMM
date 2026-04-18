# Fleet Research: DeepGEMM

**Forked:** 2026-04-18
**Source:** [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)

## Why We Forked
DeepSeek's open-source FP8 general matrix multiply kernels. Achieves 1550 TFLOPS on H800. This is the state of the art for CUDA inference — directly relevant to FM's CUDA PTX tile work and JC1's Jetson deployment.

## What to Extract
- **FP8 quantization patterns** — how they handle precision loss in matmul
- **JIT compilation** — runtime kernel compilation without pre-build
- **Fine-grained scaling** — per-block scaling factors for FP8
- **Kernel structure** — clean CUDA without heavy CUTLASS/CuTe dependencies
- **MoE support** — grouped GEMM for Mixture-of-Experts (relevant to tile routing)

## Fleet Integration Targets
- `cudaclaw` — GPU-resident agent runtime needs fast matmul
- `cuda-genepool` (JC1) — gene evaluation is essentially matmul, FP8 speeds up evolution
- FM's CUDA PTX tile marketplace — DeepGEMM patterns for tile execution
- LoRA inference pipeline (FM trains → JC1 deploys) — FP8 quantization for edge

## Notes
- ~16K lines (CUDA + Python) — substantial but well-organized
- Requires SM90 (Hopper) or SM100 — won't run on FM's RTX 4050 (Ada) or JC1's Jetson (Ampere)
- The *patterns* are what we extract, not direct execution
- BF16 support in progress — may work on our hardware
- Design philosophy: simplicity over template metaprogramming — fits our fleet style
