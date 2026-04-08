from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, field_validator


class Shot(BaseModel):
    """
    A single plasma discharge (shot) from a fusion device.

    shot_id is globally unique across all devices: "{device_id}:{native_shot_id}"
    e.g. "cmod:1120815012", "mast:30420", "lhd:184255"
    """

    shot_id: str
    device_id: str
    native_shot_id: str
    timestamp: datetime | None = None
    duration_s: float | None = None
    plasma_current_MA: float | None = None
    line_averaged_density_m3: float | None = None
    total_input_power_MW: float | None = None
    tags: list[str] = []
    status: Literal["good", "disrupted", "incomplete", "unknown"] = "unknown"

    @field_validator("shot_id")
    @classmethod
    def shot_id_format(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("shot_id must be in the format '{device_id}:{native_shot_id}'")
        return v

    @property
    def device_prefix(self) -> str:
        return self.shot_id.split(":")[0]


class ShotMetadata(Shot):
    """Extended shot metadata including machine-specific extras."""

    additional_fields: dict[str, Any] = {}
    diagnostic_count: int = 0
    license: str | None = None


class ShotSearchParams(BaseModel):
    """Parameters for searching shots across a device."""

    device_id: str
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_plasma_current_MA: float | None = None
    max_plasma_current_MA: float | None = None
    min_density_m3: float | None = None
    max_density_m3: float | None = None
    min_duration_s: float | None = None
    status: Literal["good", "disrupted", "incomplete", "unknown"] | None = None
    limit: int = 50
    offset: int = 0
