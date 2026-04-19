from pydantic import BaseModel


class Signal(BaseModel):
    """
    A time-series signal retrieved from a fusion device diagnostic.

    time_s and data are plain Python lists to ensure JSON serializability.
    For 2-D data (e.g. profiles), data is a flat list and shape is provided
    so the caller can reshape: np.array(signal.data).reshape(signal.shape)
    """

    shot_id: str
    diagnostic: str
    native_name: str
    units: str | None = None
    time_s: list[float]
    data: list[float | None] | list[list[float | None]]
    shape: list[int]
    sample_rate_hz: float | None = None
    downsampled: bool = False
    original_n_samples: int | None = None
    n_null_samples: int = 0
    metadata: dict = {}


class SignalSummary(BaseModel):
    """
    Statistical summary of a signal — returned without loading the full time series.
    Use this before get_signal to understand what you're about to retrieve.
    """

    shot_id: str
    diagnostic: str
    native_name: str
    units: str | None = None
    n_samples: int
    duration_s: float
    t_start_s: float
    t_end_s: float
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    has_gaps: bool = False
    n_null_samples: int = 0
    sparkline: str | None = None
