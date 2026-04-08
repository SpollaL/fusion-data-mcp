"""Server configuration via environment variables."""

from __future__ import annotations

import os
from typing import Literal


class Config:
    # Transport: "stdio" for local Claude Desktop use, "sse" for remote
    transport: Literal["stdio", "sse"] = os.getenv("MCP_TRANSPORT", "stdio")  # type: ignore

    # MDSplus connection pool size for C-Mod
    cmod_pool_size: int = int(os.getenv("CMOD_POOL_SIZE", "3"))

    # Default max samples returned by get_signal
    default_max_samples: int = int(os.getenv("DEFAULT_MAX_SAMPLES", "10000"))

    # Enable/disable individual connectors
    enable_cmod: bool = os.getenv("ENABLE_CMOD", "true").lower() == "true"
    enable_mast: bool = os.getenv("ENABLE_MAST", "true").lower() == "true"
    enable_lhd: bool = os.getenv("ENABLE_LHD", "true").lower() == "true"


config = Config()
