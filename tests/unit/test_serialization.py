"""
Unit tests for serialization.py.

Focus: NaN/Inf handling, downsampling correctness, numpy→JSON safety.
We don't test that Python lists are lists.
"""

import json
import math

import numpy as np
import pytest

from fusion_data_mcp.serialization import (
    _array_to_list,
    _count_nulls,
    _has_gaps,
    downsample,
    signal_from_arrays,
    summary_from_arrays,
    to_json_safe,
)


# ------------------------------------------------------------------ #
# NaN / Inf → None                                                     #
# ------------------------------------------------------------------ #

class TestArrayToList:
    def test_nan_becomes_none(self):
        arr = np.array([1.0, np.nan, 3.0])
        result = _array_to_list(arr)
        assert result[1] is None
        assert result[0] == 1.0
        assert result[2] == 3.0

    def test_pos_inf_becomes_none(self):
        arr = np.array([np.inf, 1.0])
        assert _array_to_list(arr)[0] is None

    def test_neg_inf_becomes_none(self):
        arr = np.array([-np.inf, 1.0])
        assert _array_to_list(arr)[0] is None

    def test_2d_array_preserves_shape(self):
        arr = np.ones((3, 4))
        result = _array_to_list(arr)
        assert len(result) == 3
        assert all(len(row) == 4 for row in result)

    def test_2d_nan_becomes_none(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = _array_to_list(arr)
        assert result[0][1] is None


class TestCountNulls:
    def test_clean_array_is_zero(self, clean_data):
        assert _count_nulls(clean_data) == 0

    def test_counts_nan_and_inf(self, dirty_data):
        # dirty_data has nan at 10, +inf at 50, -inf at 100 → 3 non-finite
        assert _count_nulls(dirty_data) == 3


class TestHasGaps:
    def test_uniform_has_no_gap(self, uniform_time):
        assert _has_gaps(uniform_time) is False

    def test_large_jump_detected(self):
        t = np.array([0.0, 0.1, 0.2, 1.5, 1.6])  # gap between 0.2 and 1.5
        assert _has_gaps(t) is True

    def test_short_array_never_has_gap(self):
        assert _has_gaps(np.array([0.0, 1.0])) is False


# ------------------------------------------------------------------ #
# Downsampling                                                         #
# ------------------------------------------------------------------ #

class TestDownsample:
    def test_below_limit_unchanged(self, uniform_time, clean_data):
        t_out, d_out = downsample(uniform_time, clean_data, max_samples=2000)
        assert len(t_out) == len(uniform_time)

    def test_output_length_respected(self, uniform_time, clean_data):
        t_out, _ = downsample(uniform_time, clean_data, max_samples=100)
        assert len(t_out) == 100

    def test_first_sample_preserved(self, uniform_time, clean_data):
        t_out, d_out = downsample(uniform_time, clean_data, max_samples=100)
        assert t_out[0] == pytest.approx(uniform_time[0])
        assert d_out[0] == pytest.approx(clean_data[0])

    def test_last_sample_preserved(self, uniform_time, clean_data):
        t_out, d_out = downsample(uniform_time, clean_data, max_samples=100)
        assert t_out[-1] == pytest.approx(uniform_time[-1])
        assert d_out[-1] == pytest.approx(clean_data[-1])

    def test_2d_data_shape_preserved(self, uniform_time):
        data_2d = np.ones((1000, 5))
        _, d_out = downsample(uniform_time, data_2d, max_samples=100)
        assert d_out.shape == (100, 5)


# ------------------------------------------------------------------ #
# to_json_safe                                                         #
# ------------------------------------------------------------------ #

class TestToJsonSafe:
    def test_numpy_int_becomes_python_int(self):
        result = to_json_safe(np.int64(42))
        assert isinstance(result, int)

    def test_numpy_float_nan_becomes_none(self):
        assert to_json_safe(np.float64(np.nan)) is None

    def test_plain_float_nan_becomes_none(self):
        assert to_json_safe(float("nan")) is None

    def test_plain_float_inf_becomes_none(self):
        assert to_json_safe(float("inf")) is None

    def test_ndarray_serialized(self):
        result = to_json_safe(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]

    def test_nested_dict_with_numpy(self):
        d = {"a": np.int32(1), "b": np.array([1.0, np.nan])}
        result = to_json_safe(d)
        # Must be fully JSON-serializable
        json.dumps(result)
        assert result["a"] == 1
        assert result["b"][1] is None

    def test_datetime64_becomes_string(self):
        result = to_json_safe(np.datetime64("2021-06-15"))
        assert isinstance(result, str)
        assert "2021" in result


# ------------------------------------------------------------------ #
# signal_from_arrays / summary_from_arrays                             #
# ------------------------------------------------------------------ #

class TestSignalFromArrays:
    def test_no_downsampling_flag_when_small(self, uniform_time, clean_data):
        sig = signal_from_arrays(
            shot_id="mast:1",
            diagnostic="plasma_current",
            native_name="amc/plasma_current",
            time_arr=uniform_time,
            data_arr=clean_data,
            max_samples=10_000,
        )
        assert sig.downsampled is False
        assert sig.original_n_samples is None

    def test_downsampling_flag_set_when_large(self):
        t = np.linspace(0, 10, 50_000)
        d = np.ones(50_000)
        sig = signal_from_arrays(
            shot_id="mast:1",
            diagnostic="plasma_current",
            native_name="amc/plasma_current",
            time_arr=t,
            data_arr=d,
            max_samples=10_000,
        )
        assert sig.downsampled is True
        assert sig.original_n_samples == 50_000
        assert len(sig.time_s) == 10_000

    def test_null_count_propagated(self, uniform_time, dirty_data):
        sig = signal_from_arrays(
            shot_id="mast:1",
            diagnostic="plasma_current",
            native_name="amc/plasma_current",
            time_arr=uniform_time,
            data_arr=dirty_data,
        )
        assert sig.n_null_samples == 3

    def test_model_dump_is_json_safe(self, uniform_time, clean_data):
        sig = signal_from_arrays(
            shot_id="mast:1",
            diagnostic="plasma_current",
            native_name="amc/plasma_current",
            time_arr=uniform_time,
            data_arr=clean_data,
        )
        # Must not raise — no numpy types must leak
        json.dumps(sig.model_dump())


class TestSummaryFromArrays:
    def test_stats_correct(self):
        t = np.linspace(0, 1, 100)
        d = np.arange(100.0)
        s = summary_from_arrays(
            shot_id="lhd:1",
            diagnostic="plasma_current",
            native_name="magnetics/ip",
            time_arr=t,
            data_arr=d,
        )
        assert s.min == pytest.approx(0.0)
        assert s.max == pytest.approx(99.0)
        assert s.mean == pytest.approx(49.5)
        assert s.n_samples == 100

    def test_nans_excluded_from_stats(self, uniform_time, dirty_data):
        s = summary_from_arrays(
            shot_id="lhd:1",
            diagnostic="plasma_current",
            native_name="magnetics/ip",
            time_arr=uniform_time,
            data_arr=dirty_data,
        )
        # Mean should not be NaN even with non-finite values in data
        assert s.mean is not None
        assert not math.isnan(s.mean)

    def test_has_gaps_reflected(self):
        t = np.array([0.0, 0.1, 0.2, 1.5, 1.6])
        d = np.ones(5)
        s = summary_from_arrays(
            shot_id="lhd:1", diagnostic="x", native_name="x", time_arr=t, data_arr=d
        )
        assert s.has_gaps is True
