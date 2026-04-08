"""Abstract base class for all fusion data source connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
