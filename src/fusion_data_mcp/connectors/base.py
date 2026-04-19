"""Abstract base class for all fusion data source connectors."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..config import config
from ..models import (
    DeviceInfo,
    DiagnosticList,
    EquilibriumData,
    Shot,
    ShotMetadata,
    ShotSearchParams,
    Signal,
    SignalSummary,
)


class AbstractConnector(ABC):
    """
    Every data source (LHD, MAST, C-Mod, ...) implements this interface.
    All methods are async; blocking I/O must be wrapped with asyncio.to_thread().
    """

    @property
    @abstractmethod
    def device_id(self) -> str:
        """Canonical machine identifier, e.g. 'lhd', 'mast', 'cmod'."""

    @property
    @abstractmethod
    def device_info(self) -> DeviceInfo:
        """Static metadata describing this device."""

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def search_shots(self, params: ShotSearchParams) -> list[Shot]:
        """Return shots matching the given search parameters."""

    @abstractmethod
    async def get_shot_metadata(self, shot_id: str) -> ShotMetadata:
        """Return full metadata for a specific shot (canonical or native ID)."""

    @abstractmethod
    async def list_diagnostics(self, shot_id: str | None = None) -> DiagnosticList:
        """
        List diagnostics available for this device.
        If shot_id is given, filter to diagnostics available for that shot.
        """

    # ------------------------------------------------------------------ #
    # Data retrieval                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_signal(
        self,
        shot_id: str,
        diagnostic: str,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        max_samples: int = 10_000,
    ) -> Signal:
        """
        Retrieve a signal time series.
        Accepts both canonical diagnostic names and native names.
        """

    @abstractmethod
    async def describe_signal(self, shot_id: str, diagnostic: str) -> SignalSummary:
        """
        Return a statistical summary of a signal without loading full data.
        Use this before get_signal to understand size and quality.
        """

    @abstractmethod
    async def get_equilibrium(self, shot_id: str) -> EquilibriumData | None:
        """
        Return MHD equilibrium reconstruction data, or None if unavailable.
        """

    async def download_signal(
        self,
        shot_id: str,
        diagnostic: str,
        *,
        output_dir: Path | None = None,
        fmt: str = "npz",
    ) -> dict:
        """
        Download a full-resolution signal to a local file and return metadata.

        Fetches the complete time series (no downsampling) and writes it to
        `output_dir` (default: ~/.cache/fusion-data/). Returns the file path
        and basic metadata — no data arrays in the response.
        """
        signal = await self.get_signal(shot_id, diagnostic, max_samples=10_000_000)

        safe_name = f"{shot_id.replace(':', '_')}_{diagnostic}.{fmt}"
        dest_dir = output_dir or config.download_dir
        path = dest_dir / safe_name

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            time_arr = np.array(signal.time_s)
            data_arr = np.array(signal.data)
            if fmt == "npz":
                np.savez(path, time_s=time_arr, data=data_arr)
            elif fmt == "csv":
                import csv
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["time_s", "data"])
                    for t, d in zip(time_arr.tolist(), data_arr.tolist()):
                        writer.writerow([t, d])

        await asyncio.to_thread(_write)

        n_samples = signal.original_n_samples or len(signal.time_s)
        duration = signal.time_s[-1] - signal.time_s[0] if signal.time_s else 0.0

        return {
            "path": str(path),
            "shot_id": shot_id,
            "diagnostic": diagnostic,
            "native_name": signal.native_name,
            "units": signal.units,
            "format": fmt,
            "n_samples": n_samples,
            "duration_s": round(duration, 6),
            "file_size_mb": round(path.stat().st_size / 1e6, 3),
        }

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable."""

    async def close(self) -> None:
        """Release connections and file handles. Override if needed."""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def make_shot_id(self, native_id: str | int) -> str:
        """Build a canonical shot_id from a native shot identifier."""
        return f"{self.device_id}:{native_id}"

    def parse_native_id(self, shot_id: str) -> str:
        """Extract the native shot identifier from a canonical shot_id."""
        if ":" in shot_id:
            return shot_id.split(":", 1)[1]
        return shot_id
