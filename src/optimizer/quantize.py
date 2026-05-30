"""
Quantization strategies for LLM inference on AMD GPUs.

Implements GPTQ and AWQ quantization with ROCm-optimized dequantization
kernels for low-precision inference on MI200/MI300 series accelerators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class QuantConfig:
    """Configuration for quantization passes."""

    bits: int = 4
    group_size: int = 128
    sym: bool = True
    desc_act: bool = False
    use_rocm_kernels: bool = True
    calibration_samples: int = 128
    calibration_seqlen: int = 2048
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 8):
            raise ValueError(f"Unsupported bit width: {self.bits}")
        if self.group_size <= 0 or self.group_size % 16 != 0:
            raise ValueError(
                f"group_size must be positive and divisible by 16, got {self.group_size}"
            )


class BaseQuantizer:
    """Base class shared by GPTQ and AWQ quantizers."""

    def __init__(self, config: QuantConfig) -> None:
        self.config = config
        self._scale_dtype = "float16"

    def quantize(self, model: Any, calibration_data: Iterator[Any] | None = None) -> Any:
        """Quantize a model in-place and return the modified module.

        Args:
            model: A PyTorch ``nn.Module`` to quantize.
            calibration_data: Iterable of calibration tensors.

        Returns:
            The quantized model with packed weight tensors.
        """
        raise NotImplementedError

    def save(self, model: Any, output_dir: str | Path) -> None:
        """Persist quantized weights and metadata to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving quantized model to %s", output_dir)
        meta = {
            "bits": self.config.bits,
            "group_size": self.config.group_size,
            "sym": self.config.sym,
            "scheme": self.__class__.__name__,
            "rocm_kernels": self.config.use_rocm_kernels,
        }
        (output_dir / "quant_config.json").write_text(_json_dumps(meta))

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _pack_weights(self, weight, scales, zeros):
        """Pack quantized weights into the storage layout expected by ROCm kernels."""
        raise NotImplementedError


class GPTQQuantizer(BaseQuantizer):
    """GPTQ quantization with optional ROCm-fused dequant kernels.

    GPTQ uses second-order information from a calibration dataset to
    minimise the layer-wise quantization error. Packing follows the
    ``int4 -> uint32`` interleaved layout expected by the bundled
    ``hipblaslt`` kernels.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        use_rocm_kernels: bool = True,
    ) -> None:
        super().__init__(
            QuantConfig(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,
                use_rocm_kernels=use_rocm_kernels,
            )
        )

    def quantize(self, model, calibration_data=None):
        logger.info(
            "Running GPTQ %d-bit quantization (group_size=%d, desc_act=%s)",
            self.config.bits,
            self.config.group_size,
            self.config.desc_act,
        )
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for GPTQ quantization") from exc

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                self._quantize_layer(name, module)
        return model

    def _quantize_layer(self, name, module):
        logger.debug("Quantizing layer %s", name)
        # In production this calls the C++/HIP extension. We expose a
        # Python-friendly stub here that reshapes weights into groups,
        # computes scales/zero-points, and packs them.
        weight = module.weight.data
        groups = weight.reshape(weight.shape[0], -1, self.config.group_size)
        scales = groups.abs().amax(dim=-1, keepdim=True) / ((1 << (self.config.bits - 1)) - 1)
        zeros = None if self.config.sym else groups.mean(dim=-1, keepdim=True)
        module.weight = self._pack_weights(weight, scales, zeros)
        module._gptq_scales = scales
        module._gptq_zeros = zeros

    def _pack_weights(self, weight, scales, zeros):
        # Place-holder pack: real implementation lives in the hip extension.
        return weight


class AWQQuantizer(BaseQuantizer):
    """Activation-aware Weight Quantization for ROCm.

    AWQ scales weights along the output channel dimension based on
    activation magnitudes observed during calibration, preserving
    important channels under aggressive 4-bit quantization.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        use_rocm_kernels: bool = True,
    ) -> None:
        super().__init__(
            QuantConfig(
                bits=bits,
                group_size=group_size,
                sym=not zero_point,
                use_rocm_kernels=use_rocm_kernels,
            )
        )
        self.zero_point = zero_point

    def quantize(self, model, calibration_data=None):
        logger.info(
            "Running AWQ %d-bit quantization (zero_point=%s)",
            self.config.bits,
            self.zero_point,
        )
        # Calibration pass: collect activation statistics.
        scales = self._collect_activation_scales(model, calibration_data)
        return self._apply_scales(model, scales)

    def _collect_activation_scales(self, model, calibration_data):
        scales: dict[str, Any] = {}
        if calibration_data is None:
            logger.warning("No calibration data supplied; using uniform scales")
            return scales
        for batch in calibration_data:
            # Hook each linear layer and record max-abs activations.
            _ = batch  # placeholder for the activation hook call
        return scales

    def _apply_scales(self, model, scales):
        return model


def _json_dumps(obj: dict[str, Any]) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)
