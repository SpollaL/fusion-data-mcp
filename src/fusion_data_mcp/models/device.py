from typing import Literal

from pydantic import BaseModel


class DeviceCapabilities(BaseModel):
    has_equilibrium: bool
    searchable_by_date: bool
    searchable_by_plasma_params: bool
    max_signal_duration_s: float | None = None


class DeviceInfo(BaseModel):
    id: str
    name: str
    country: str
    type: Literal["tokamak", "stellarator", "other"]
    description: str
    capabilities: DeviceCapabilities
    data_license: str | None = None
    data_url: str | None = None
