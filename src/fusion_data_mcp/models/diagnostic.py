from typing import Literal

from pydantic import BaseModel

# Canonical diagnostic names shared across machines.
# Each connector maps its native names to these where possible.
CANONICAL_DIAGNOSTICS: dict[str, dict[str, str]] = {
    "plasma_current": {
        "description": "Total plasma current",
        "units": "MA",
        "category": "magnetics",
        "signal_type": "scalar",
    },
    "electron_temperature": {
        "description": "Electron temperature (core or profile)",
        "units": "keV",
        "category": "thomson_scattering",
        "signal_type": "profile",
    },
    "electron_density": {
        "description": "Electron density (line-averaged or profile)",
        "units": "m^-3",
        "category": "thomson_scattering",
        "signal_type": "profile",
    },
    "stored_energy": {
        "description": "Total plasma stored energy (MHD)",
        "units": "MJ",
        "category": "magnetics",
        "signal_type": "scalar",
    },
    "loop_voltage": {
        "description": "Ohmic loop voltage",
        "units": "V",
        "category": "magnetics",
        "signal_type": "scalar",
    },
    "h_alpha": {
        "description": "H-alpha emission (edge recycling indicator)",
        "units": "a.u.",
        "category": "spectroscopy",
        "signal_type": "scalar",
    },
    "neutral_beam_power": {
        "description": "Total neutral beam injection power",
        "units": "MW",
        "category": "heating",
        "signal_type": "scalar",
    },
    "rf_power": {
        "description": "Total RF heating power",
        "units": "MW",
        "category": "heating",
        "signal_type": "scalar",
    },
    "soft_xray": {
        "description": "Soft X-ray emission (MHD activity proxy)",
        "units": "a.u.",
        "category": "radiation",
        "signal_type": "scalar",
    },
    "radiated_power": {
        "description": "Total radiated power (bolometry)",
        "units": "MW",
        "category": "radiation",
        "signal_type": "scalar",
    },
}


class Diagnostic(BaseModel):
    name: str
    native_name: str
    canonical_name: str | None = None
    category: str
    description: str | None = None
    signal_type: Literal["scalar", "profile", "image", "unknown"] = "unknown"
    units: str | None = None
    available: bool = True


class DiagnosticList(BaseModel):
    device_id: str
    shot_id: str | None = None
    diagnostics: list[Diagnostic]
    total: int
