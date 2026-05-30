"""
Benchmark harness for the AMD LLM inference optimizer.

Measures throughput (tokens/sec), end-to-end latency, time-to-first-token
and peak GPU memory across different optimization configurations.

Example
-------
    python benchmarks/run_benchmark.py \\
        --model meta-llama/Llama-2-7b-hf \\
        --quantization gptq-4bit \\
        --batch-sizes 1,4,16,32 \\
        --output-json results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model: str
    quantization: str | None
    batch_size: int
    input_tokens: int
    output_tokens: int
    throughput_tok_per_s: float
    latency_ms: float
    ttft_ms: float
    peak_memory_gb: float


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LLM inference on AMD GPUs")
    parser.add_argument("--model", required=True, help="Model identifier or local path")
    parser.add_argument(
        "--quantization",
        default=None,
        choices=[None, "gptq-4bit", "gptq-8bit", "awq-4bit"],
        help="Quantization scheme (None for fp16 baseline)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,4,16,32",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default="rocm")
    return parser


def _generate_sample_prompts(batch_size: int, input_tokens: int) -> list[str]:
    base = "Explain in technical detail how transformer attention scales with sequence length. "
    repeats = max(1, input_tokens // 12)
    return [(base * repeats)[: input_tokens * 4] for _ in range(batch_size)]


def _measure_peak_memory_gb(device: str) -> float:
    try:
        import torch

        if device == "rocm" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception as exc:  # pragma: no cover - depends on runtime
        logger.debug("Failed to query peak memory: %s", exc)
    return 0.0


def run_single_config(
    engine: Any,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    warmup: int,
    repeats: int,
    device: str,
) -> BenchmarkResult:
    """Run benchmark sweep for a single (model, quant, batch_size) configuration."""
    prompts = _generate_sample_prompts(batch_size, input_tokens)

    # Warmup
    for _ in range(warmup):
        _ = engine.generate(prompts, max_tokens=output_tokens)

    latencies: list[float] = []
    ttfts: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        engine.generate(prompts, max_tokens=output_tokens)
        latencies.append((time.perf_counter() - start) * 1000)
        ttfts.append(latencies[-1] / output_tokens)  # simple proxy

    median_latency = statistics.median(latencies)
    total_out = batch_size * output_tokens
    throughput = total_out / (median_latency / 1000)

    return BenchmarkResult(
        model=getattr(engine, "model_name", "unknown"),
        quantization=getattr(engine, "quantization", None),
        batch_size=batch_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        throughput_tok_per_s=round(throughput, 2),
        latency_ms=round(median_latency, 2),
        ttft_ms=round(statistics.median(ttfts), 2),
        peak_memory_gb=round(_measure_peak_memory_gb(device), 2),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from optimizer import InferenceEngine  # noqa: WPS433

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    results: list[BenchmarkResult] = []

    for bs in batch_sizes:
        logger.info("Benchmarking %s @ bs=%d quant=%s", args.model, bs, args.quantization)
        engine = InferenceEngine(
            model_name=args.model,
            quantization=args.quantization,
            max_batch_size=bs,
            device=args.device,
        ).load()
        result = run_single_config(
            engine,
            batch_size=bs,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
            device=args.device,
        )
        results.append(result)
        logger.info(
            "bs=%d  throughput=%.1f tok/s  latency=%.1fms  mem=%.2fGB",
            bs,
            result.throughput_tok_per_s,
            result.latency_ms,
            result.peak_memory_gb,
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps([asdict(r) for r in results], indent=2))
        logger.info("Wrote results to %s", args.output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
