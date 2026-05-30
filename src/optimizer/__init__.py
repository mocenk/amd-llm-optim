"""
AMD LLM Inference Optimizer
============================

A high-performance toolkit for optimizing LLM inference on AMD GPUs
using ROCm, featuring quantization, dynamic batching, and KV-cache
optimization.
"""

__version__ = "0.3.0"
__author__ = "AMD LLM Optim Contributors"

from optimizer.quantize import GPTQQuantizer, AWQQuantizer
from optimizer.batch_engine import ContinuousBatchEngine
from optimizer.kv_cache import PagedKVCache
from optimizer.rocm_kernels import ROCmKernelLauncher


class InferenceEngine:
    """Main entry point for optimized LLM inference on AMD GPUs.

    Combines quantization, batching, and memory optimization into a
    unified inference pipeline.

    Args:
        model_name: HuggingFace model identifier or local path.
        quantization: Quantization strategy ('gptq-4bit', 'awq-4bit', None).
        kv_cache_pages: Number of pages for KV-cache memory pool.
        max_batch_size: Maximum concurrent sequences in a batch.
        device: Target device ('rocm', 'cuda', 'cpu').
    """

    def __init__(
        self,
        model_name: str,
        quantization: str | None = "gptq-4bit",
        kv_cache_pages: int = 2048,
        max_batch_size: int = 64,
        device: str = "rocm",
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.max_batch_size = max_batch_size
        self.device = device

        self._kernel_launcher = ROCmKernelLauncher(device=device)
        self._kv_cache = PagedKVCache(
            num_pages=kv_cache_pages,
            page_size=16,
            num_heads=32,
            head_dim=128,
        )
        self._batch_engine = ContinuousBatchEngine(
            max_batch_size=max_batch_size,
            kv_cache=self._kv_cache,
        )
        self._model = None
        self._tokenizer = None

    def load(self) -> "InferenceEngine":
        """Load and optimize the model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        if self.quantization:
            quantizer = self._get_quantizer()
            self._model = quantizer.quantize(self._model)

        return self

    def _get_quantizer(self):
        """Get the appropriate quantizer based on config."""
        if self.quantization.startswith("gptq"):
            bits = int(self.quantization.split("-")[1].replace("bit", ""))
            return GPTQQuantizer(bits=bits, use_rocm_kernels=True)
        elif self.quantization.startswith("awq"):
            bits = int(self.quantization.split("-")[1].replace("bit", ""))
            return AWQQuantizer(bits=bits, use_rocm_kernels=True)
        raise ValueError(f"Unknown quantization: {self.quantization}")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: Input text prompts.
            max_tokens: Maximum tokens to generate per prompt.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of generated text completions.
        """
        if self._model is None:
            self.load()

        results = self._batch_engine.submit(
            prompts=prompts,
            tokenizer=self._tokenizer,
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return results
