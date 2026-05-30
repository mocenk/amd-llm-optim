"""
ROCm/HIP kernel launchers and AMD GPU utilities.

Wraps the small set of HIP entry points used by the inference engine:
GPU detection, stream management, and kernel launches for fused
attention, dequant-GEMM, and layer-norm.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUInfo:
    """Identifying info for an AMD GPU."""

    index: int
    name: str
    arch: str
    vram_gb: float
    compute_units: int


class ROCmKernelLauncher:
    """Thin wrapper around HIP streams and custom ROCm kernels."""

    def __init__(self, device: str = "rocm", num_streams: int = 4) -> None:
        self.device = device
        self.num_streams = num_streams
        self._streams: list = []
        self._gpus: list[GPUInfo] = []
        if device == "rocm":
            self._init_rocm()

    def _init_rocm(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("PyTorch not available; ROCm launcher running in dry-run mode")
            return
        self._gpus = self.detect_gpus()
        logger.info("Detected %d AMD GPU(s): %s", len(self._gpus), [g.name for g in self._gpus])

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_gpus() -> list[GPUInfo]:
        """Detect AMD GPUs via rocm-smi, falling back gracefully."""
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            logger.debug("rocm-smi unavailable: %s", exc)
            return []
        gpus: list[GPUInfo] = []
        for idx, line in enumerate(out.strip().splitlines()[1:]):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name = parts[1] if len(parts) > 1 else f"AMD GPU {idx}"
            arch = ROCmKernelLauncher._infer_arch(name)
            vram_gb = ROCmKernelLauncher._parse_vram(parts[-1])
            gpus.append(GPUInfo(index=idx, name=name, arch=arch, vram_gb=vram_gb, compute_units=0))
        return gpus

    @staticmethod
    def _infer_arch(name: str) -> str:
        n = name.lower()
        if "mi300" in n:
            return "gfx942"
        if "mi250" in n:
            return "gfx90a"
        if "mi210" in n:
            return "gfx90a"
        if "7900" in n:
            return "gfx1100"
        return "unknown"

    @staticmethod
    def _parse_vram(text: str) -> float:
        try:
            value = float("".join(c for c in text if c.isdigit() or c == "."))
        except ValueError:
            return 0.0
        if "MiB" in text or "MB" in text:
            return value / 1024.0
        return value

    # ------------------------------------------------------------------
    # Kernel launches (delegate to compiled HIP extension when available)
    # ------------------------------------------------------------------
    def fused_attention(self, q, k, v, mask=None):
        """Launch the fused flash-attention kernel."""
        ext = _load_hip_extension()
        if ext is None:
            return _torch_attention_fallback(q, k, v, mask)
        return ext.fused_attention(q, k, v, mask)

    def dequant_gemm(self, x, packed_w, scales, zeros, bits: int = 4):
        """Fused dequant + GEMM for GPTQ/AWQ packed weights."""
        ext = _load_hip_extension()
        if ext is None:
            return _torch_dequant_fallback(x, packed_w, scales, zeros, bits)
        return ext.dequant_gemm(x, packed_w, scales, zeros, bits)

    def rms_norm(self, x, weight, eps: float = 1e-6):
        ext = _load_hip_extension()
        if ext is None:
            return _torch_rms_norm_fallback(x, weight, eps)
        return ext.rms_norm(x, weight, eps)


_HIP_EXTENSION = None


def _load_hip_extension():
    global _HIP_EXTENSION
    if _HIP_EXTENSION is not None:
        return _HIP_EXTENSION
    if os.environ.get("AMD_LLM_DISABLE_HIP") == "1":
        return None
    try:
        import optimizer._hip_ext as ext  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("HIP extension not built; using torch fallbacks")
        return None
    _HIP_EXTENSION = ext
    return ext


def _torch_attention_fallback(q, k, v, mask):
    import torch
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _torch_dequant_fallback(x, packed_w, scales, zeros, bits):
    import torch

    weight = packed_w.to(torch.float16) * scales
    if zeros is not None:
        weight = weight - zeros
    return x @ weight.t()


def _torch_rms_norm_fallback(x, weight, eps):
    import torch

    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight
