# Architecture overview

This document sketches the high-level pipeline of the AMD LLM optimizer
and the reasoning behind each subsystem.

## Pipeline

```
                ┌─────────────────────┐
   prompts ───▶│  ContinuousBatch    │──▶ scheduled batch
                │  Engine             │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  PagedKVCache       │  block-wise attention memory
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Quantized Linear   │  GPTQ / AWQ + HIP dequant kernel
                │  (ROCm kernels)     │
                └──────────┬──────────┘
                           │
                           ▼
                       generated tokens
```

## Why these choices

### Continuous batching
Static batching wastes GPU cycles when sequences finish at different
times. Continuous batching pulls in new requests as soon as a slot
frees, keeping the SMs saturated. We schedule on a 5 ms quantum which
is short enough to feel real-time and long enough to amortize launch
overhead on CDNA / RDNA.

### Paged KV-cache
A flat KV-cache forces worst-case memory allocation per sequence. The
paged layout (16-token pages, like vLLM) lets us pack many sequences
into the same HBM budget and reuse pages across prefix-shared prompts.

### Quantization
- **GPTQ** — second-order calibrated 4-bit, best preservation of
  perplexity at the cost of a slow offline calibration pass.
- **AWQ** — activation-aware 4-bit, better when calibration data is
  scarce or representative-distribution sensitive.

Both share an ``int4 → uint32`` interleaved storage so the same HIP
dequant kernel handles both.

### ROCm kernels
We prefer ``hipblaslt`` for matmul and a custom paged-attention HIP
kernel for the attention block. ``num_streams=4`` overlaps prefill with
in-flight decode; ``graph_capture=true`` reuses launch graphs once the
batch shape stabilizes.

## Hardware targets

| Arch    | Card examples           | Status         |
| ------- | ----------------------- | -------------- |
| CDNA2   | MI210, MI250X           | Supported      |
| CDNA3   | MI300A, MI300X          | Supported      |
| RDNA3   | W7900, W7800, RX 7900   | Experimental   |
| RDNA4   | (next-gen)              | Planned        |

## Future work

- Speculative decoding with a draft head
- FP8 (E4M3 / E5M2) on MI300
- Multi-LoRA serving with adapter swapping in cache
- Disaggregated prefill / decode across two GPUs
