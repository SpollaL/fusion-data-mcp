"""
Converts scientific Python types (numpy, xarray) to JSON-safe Python primitives.

All connectors must pass their data through this layer before returning models.
The key invariant: nothing that enters a Pydantic model should be a numpy type.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models.signal import Signal, SignalSummary

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


def _scalar(v: Any) -> float | None:
    """Convert a numpy scalar or Python number to a JSON-safe float."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _array_to_list(arr: Any) -> list:
    """
    Convert a numpy array (1-D or 2-D) to a nested Python list.
    NaN and ±Inf are replaced with None.
    """
    if isinstance(arr, list):
        return arr
    arr = np.asarray(arr, dtype=float)
    # Replace non-finite values with np.nan, then mask to None via object cast
    finite = np.where(np.isfinite(arr), arr, np.nan)
    if finite.ndim == 1:
        return [None if math.isnan(x) else float(x) for x in finite.tolist()]
    elif finite.ndim == 2:
        return [
            [None if math.isnan(x) else float(x) for x in row]
            for row in finite.tolist()
        ]
    else:
        # Flatten higher-dimensional arrays
        return _array_to_list(finite.reshape(-1))


def _count_nulls(arr: Any) -> int:
    """Count NaN/Inf values in a numpy array."""
    a = np.asarray(arr, dtype=float)
    return int(np.sum(~np.isfinite(a)))


def _has_gaps(time_arr: Any, rtol: float = 0.1) -> bool:
    """
    Detect non-uniform sampling in a time array.
    A gap is any dt more than rtol*median_dt larger than the median.
    """
    t = np.asarray(time_arr, dtype=float)
    if len(t) < 3:
        return False
    dt = np.diff(t)
    median_dt = np.median(dt)
    if median_dt <= 0:
        return False
    return bool(np.any(dt > (1 + rtol) * median_dt))


_SPARKS = "▁▂▃▄▅▆▇█"


def generate_sparkline(data_arr: np.ndarray, width: int = 40) -> str:
    """
    Generate an ASCII sparkline from a 1-D numpy array.

    Buckets the data into `width` equal windows, takes the mean of each
    (ignoring NaN/Inf), and maps to Unicode block characters ▁–█.
    Returns an empty string if the data is all-NaN or has fewer than 2 values.
    """
    arr = np.asarray(data_arr, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        return ""

    # Bucket into `width` windows
    bucket_means = []
    indices = np.array_split(np.arange(len(arr)), width)
    for idx in indices:
        chunk = arr[idx]
        f = chunk[np.isfinite(chunk)]
        bucket_means.append(float(np.mean(f)) if len(f) > 0 else float("nan"))

    bucket_means = np.array(bucket_means)
    finite_means = bucket_means[np.isfinite(bucket_means)]
    if len(finite_means) == 0:
        return ""

    lo, hi = finite_means.min(), finite_means.max()
    if hi == lo:
        # Flat signal — use mid-level character
        return _SPARKS[3] * width

    chars = []
    for v in bucket_means:
        if not math.isfinite(v):
            chars.append(" ")
        else:
            level = int((v - lo) / (hi - lo) * (len(_SPARKS) - 1))
            chars.append(_SPARKS[level])

    return "".join(chars)


def downsample(time_arr: np.ndarray, data_arr: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform stride downsampling. Preserves first and last sample."""
    n = len(time_arr)
    if n <= max_samples:
        return time_arr, data_arr
    indices = np.round(np.linspace(0, n - 1, max_samples)).astype(int)
    return time_arr[indices], data_arr[indices] if data_arr.ndim == 1 else data_arr[indices, :]


def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert scientific Python types to JSON-safe equivalents.
    Use for ad-hoc dicts (e.g. metadata) before passing to Pydantic models.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _array_to_list(obj)
    if HAS_XARRAY and isinstance(obj, xr.DataArray):
        return _array_to_list(obj.values)
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    # datetime64 → ISO string
    if isinstance(obj, np.datetime64):
        return str(obj)
    return str(obj)


def signal_from_arrays(
    *,
    shot_id: str,
    diagnostic: str,
    native_name: str,
    time_arr: np.ndarray,
    data_arr: np.ndarray,
    units: str | None = None,
    max_samples: int = 10_000,
    metadata: dict | None = None,
) -> Signal:
    """
    Build a Signal model from raw numpy arrays.
    Applies downsampling, null counting, and JSON-safe conversion.
    """
    original_n = len(time_arr)
    n_nulls = _count_nulls(data_arr)

    time_arr, data_arr = downsample(time_arr, data_arr, max_samples)
    downsampled = len(time_arr) < original_n

    dt = np.diff(time_arr)
    sample_rate = float(1.0 / np.median(dt)) if len(dt) > 0 and np.median(dt) > 0 else None

    return Signal(
        shot_id=shot_id,
        diagnostic=diagnostic,
        native_name=native_name,
        units=units,
        time_s=_array_to_list(time_arr),
        data=_array_to_list(data_arr),
        shape=list(data_arr.shape),
        sample_rate_hz=sample_rate,
        downsampled=downsampled,
        original_n_samples=original_n if downsampled else None,
        n_null_samples=n_nulls,
        metadata=to_json_safe(metadata or {}),
    )


def summary_from_arrays(
    *,
    shot_id: str,
    diagnostic: str,
    native_name: str,
    time_arr: np.ndarray,
    data_arr: np.ndarray,
    units: str | None = None,
) -> SignalSummary:
    """
    Build a SignalSummary from raw numpy arrays without downsampling.
    Safe to call on very large arrays — only computes aggregates.
    """
    finite_data = data_arr[np.isfinite(data_arr)] if data_arr.ndim == 1 else None
    n_nulls = _count_nulls(data_arr)

    return SignalSummary(
        shot_id=shot_id,
        diagnostic=diagnostic,
        native_name=native_name,
        units=units,
        n_samples=len(time_arr),
        duration_s=float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0,
        t_start_s=float(time_arr[0]) if len(time_arr) > 0 else 0.0,
        t_end_s=float(time_arr[-1]) if len(time_arr) > 0 else 0.0,
        min=float(np.min(finite_data)) if finite_data is not None and len(finite_data) > 0 else None,
        max=float(np.max(finite_data)) if finite_data is not None and len(finite_data) > 0 else None,
        mean=float(np.mean(finite_data)) if finite_data is not None and len(finite_data) > 0 else None,
        std=float(np.std(finite_data)) if finite_data is not None and len(finite_data) > 0 else None,
        has_gaps=_has_gaps(time_arr),
        n_null_samples=n_nulls,
        sparkline=generate_sparkline(data_arr) if data_arr.ndim == 1 else None,
    )
