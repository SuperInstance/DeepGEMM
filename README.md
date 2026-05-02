# DeepGEMM

DeepGEMM is a unified, high-performance tensor core kernel library that brings together the key computation primitives of modern large language models — GEMMs (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning indexer, HyperConnection (HC), and more — into a single, cohesive CUDA codebase. All kernels are compiled at runtime via a lightweight Just-In-Time (JIT) module, requiring no CUDA compilation during installation.

## Brand Line
> High-performance GEMM kernels for GPU compute — DeepSeek's gift to the Cocapn fleet.

## Installation

```bash
git clone --recursive git@github.com:SuperInstance/DeepGEMM.git
cd DeepGEMM
./develop.sh
./install.sh
```

Then import `deep_gemm` in your Python project.

## Usage

```python
import torch
import deep_gemm

# FP8 GEMM (NT layout)
D = deep_gemm.fp8_gemm_nt(A, B, D, As, Bs)
```

## Fleet Context

Part of the Cocapn fleet. DeepGEMM provides the GPU compute layer for fleet compute-intensive workloads.
Related repos:
- [JetsonClaw1-vessel](https://github.com/Lucineer/JetsonClaw1-vessel) — edge-native agent case study
- [AIR](https://github.com/SuperInstance/AIR) — adaptive intelligence runtime
- [Equipment-Swarm-Coordinator](https://github.com/SuperInstance/Equipment-Swarm-Coordinator) — multi-agent orchestration

🦐 Cocapn fleet — lighthouse keeper architecture
