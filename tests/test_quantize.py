"""Unit tests for the quantization module."""

from __future__ import annotations

import pytest

from optimizer.quantize import AWQQuantizer, GPTQQuantizer, QuantConfig


class TestQuantConfig:
    def test_defaults(self):
        cfg = QuantConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128
        assert cfg.sym is True
        assert cfg.use_rocm_kernels is True

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_supported_bit_widths(self, bits):
        cfg = QuantConfig(bits=bits)
        assert cfg.bits == bits

    @pytest.mark.parametrize("bits", [1, 5, 6, 7, 16])
    def test_rejects_unsupported_bits(self, bits):
        with pytest.raises(ValueError, match="Unsupported bit width"):
            QuantConfig(bits=bits)

    @pytest.mark.parametrize("group_size", [0, -8, 17, 33])
    def test_rejects_invalid_group_size(self, group_size):
        with pytest.raises(ValueError, match="group_size"):
            QuantConfig(group_size=group_size)


class TestGPTQQuantizer:
    def test_constructor_defaults(self):
        q = GPTQQuantizer()
        assert q.config.bits == 4
        assert q.config.group_size == 128
        assert q.config.use_rocm_kernels is True

    def test_save_writes_metadata(self, tmp_path):
        q = GPTQQuantizer(bits=4, group_size=64)

        class _Stub:
            pass

        q.save(_Stub(), tmp_path)
        meta_file = tmp_path / "quant_config.json"
        assert meta_file.exists()
        text = meta_file.read_text()
        assert '"bits": 4' in text
        assert '"group_size": 64' in text
        assert '"scheme": "GPTQQuantizer"' in text


class TestAWQQuantizer:
    def test_constructor_defaults(self):
        q = AWQQuantizer()
        assert q.config.bits == 4
        assert q.zero_point is True
        # zero_point=True implies asymmetric quantization
        assert q.config.sym is False

    def test_zero_point_disabled(self):
        q = AWQQuantizer(zero_point=False)
        assert q.config.sym is True
