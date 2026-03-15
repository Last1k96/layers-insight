"""Tests for input_generator utility."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from backend.utils.input_generator import (
    PRECISION_MAP,
    generate_random_input,
    load_input_from_file,
    prepare_inputs,
)


class TestGenerateRandomInput:
    def test_fp32(self):
        data = generate_random_input([1, 3, 224, 224], "fp32")
        assert data.shape == (1, 3, 224, 224)
        assert data.dtype == np.float32
        assert data.min() >= 0.0
        assert data.max() <= 1.0

    def test_fp16(self):
        data = generate_random_input([2, 3], "fp16")
        assert data.shape == (2, 3)
        assert data.dtype == np.float16

    def test_i32(self):
        data = generate_random_input([4, 4], "i32")
        assert data.shape == (4, 4)
        assert data.dtype == np.int32
        assert data.min() >= 0
        assert data.max() <= 10

    def test_u8(self):
        data = generate_random_input([10], "u8")
        assert data.shape == (10,)
        assert data.dtype == np.uint8
        assert data.min() >= 0
        assert data.max() <= 10

    def test_bool(self):
        data = generate_random_input([5], "bool")
        assert data.shape == (5,)
        # Bool integers are 0 or non-zero
        assert data.dtype == np.bool_

    def test_unknown_precision_defaults_fp32(self):
        data = generate_random_input([2, 2], "unknown")
        assert data.dtype == np.float32


class TestPrecisionMap:
    def test_all_keys(self):
        expected_keys = {"fp32", "fp16", "f32", "f16", "i32", "i64", "u8", "i8", "bool"}
        assert set(PRECISION_MAP.keys()) == expected_keys

    def test_f32_alias(self):
        assert PRECISION_MAP["fp32"] == PRECISION_MAP["f32"]

    def test_f16_alias(self):
        assert PRECISION_MAP["fp16"] == PRECISION_MAP["f16"]


class TestLoadInputFromFile:
    def test_load_npy(self, tmp_path):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = tmp_path / "input.npy"
        np.save(str(path), data)

        loaded = load_input_from_file(str(path))
        np.testing.assert_array_equal(loaded, data)

    def test_load_bin(self, tmp_path):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        path = tmp_path / "input.bin"
        data.tofile(str(path))

        loaded = load_input_from_file(str(path), shape=[2, 2])
        assert loaded.shape == (2, 2)

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "input.xyz"
        path.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_input_from_file(str(path))


class TestPrepareInputs:
    def test_random_fallback(self):
        params = [{"name": "input", "shape": [1, 3, 8, 8]}]
        inputs = prepare_inputs(params, precision="fp32")
        assert "input" in inputs
        assert inputs["input"].shape == (1, 3, 8, 8)
        assert inputs["input"].dtype == np.float32

    def test_per_input_configs(self, tmp_path):
        # Save a file for config
        data = np.ones((1, 3), dtype=np.float32)
        npy_path = tmp_path / "data.npy"
        np.save(str(npy_path), data)

        params = [
            {"name": "input_a", "shape": [1, 3]},
            {"name": "input_b", "shape": [2, 2]},
        ]
        configs = [
            {"name": "input_a", "source": "file", "path": str(npy_path)},
            {"name": "input_b", "data_type": "i32"},
        ]
        inputs = prepare_inputs(params, input_configs=configs)
        np.testing.assert_array_equal(inputs["input_a"], data)
        assert inputs["input_b"].dtype == np.int32

    def test_directory_input_path(self, tmp_path):
        data = np.zeros((1, 3), dtype=np.float32)
        np.save(str(tmp_path / "input.npy"), data)

        params = [{"name": "input", "shape": [1, 3]}]
        inputs = prepare_inputs(params, input_path=str(tmp_path))
        np.testing.assert_array_equal(inputs["input"], data)
