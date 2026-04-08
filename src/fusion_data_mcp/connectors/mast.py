"""
MAST connector — FAIR-MAST REST/GraphQL API + Zarr object store.

Public API: https://mastapp.site
Data:       Zarr format on public S3 (no credentials required)
License:    CC BY-SA 4.0
Covers:     MAST campaigns M05–M09
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import numpy as np

import httpx

from ..models import (
    DeviceCapabilities,
    DeviceInfo,
    DiagnosticList,
    Diagnostic,
    EquilibriumData,
    Shot,
    ShotMetadata,
    ShotSearchParams,
    Signal,
    SignalSummary,
)
from ..serialization import signal_from_arrays, summary_from_arrays, to_json_safe
from .base import AbstractConnector

logger = logging.getLogger(__name__)

_BASE_URL = "https://mastapp.site"
_LICENSE = "CC BY-SA 4.0"

# Canonical → FAIR-MAST source/name mapping
_SIGNAL_MAP: dict[str, tuple[str, str]] = {
    "plasma_current":       ("amc", "plasma_current"),
    "loop_voltage":         ("amc", "loop_voltage"),
    "stored_energy":        ("efm", "wmhd"),
    "electron_density":     ("ane", "density"),
    "electron_temperature": ("ats", "electron_temperature"),
    "h_alpha":              ("ada", "dalpha_hm10"),
    "radiated_power":       ("bol", "power_radiated_total"),
    "neutral_beam_power":   ("anb", "nb_total_power"),
}


class MASTConnector(AbstractConnector):
    """MAST connector via FAIR-MAST public API."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=_BASE_URL,
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
        return self._client

    # ------------------------------------------------------------------ #
    # DeviceInfo                                                           #
    # ------------------------------------------------------------------ #

    @property
    def device_id(self) -> str:
        return "mast"

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            id="mast",
            name="Mega Amp Spherical Tokamak",
            country="UK",
            type="tokamak",
            description=(
                "Spherical tokamak operated at UKAEA Culham (2000–2013). "
                "Campaigns M05–M09 available via FAIR-MAST public API."
            ),
            capabilities=DeviceCapabilities(
                has_equilibrium=True,
                searchable_by_date=True,
                searchable_by_plasma_params=True,
            ),
            data_license=_LICENSE,
            data_url=_BASE_URL,
        )

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    async def search_shots(self, params: ShotSearchParams) -> list[Shot]:
        client = self._get_client()
        query_params: dict = {
            "limit": params.limit,
            "offset": params.offset,
        }
        if params.date_from:
            query_params["timestamp_from"] = params.date_from.isoformat()
        if params.date_to:
            query_params["timestamp_to"] = params.date_to.isoformat()
        if params.min_plasma_current_MA is not None:
            query_params["ip_min"] = params.min_plasma_current_MA * 1e6
        if params.max_plasma_current_MA is not None:
            query_params["ip_max"] = params.max_plasma_current_MA * 1e6

        try:
            resp = await client.get("/shots", params=query_params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("MAST shot search failed: %s", e)
            return []

        items = data.get("items", data) if isinstance(data, dict) else data
        shots = []
        for item in items:
            shots.append(self._parse_shot(item))
        return shots

    async def get_shot_metadata(self, shot_id: str) -> ShotMetadata:
        native_id = self.parse_native_id(shot_id)
        client = self._get_client()

        resp = await client.get(f"/shots/{native_id}")
        resp.raise_for_status()
        data = resp.json()

        shot = self._parse_shot(data)
        return ShotMetadata(
            **shot.model_dump(),
            additional_fields=to_json_safe(data),
            diagnostic_count=len(_SIGNAL_MAP),
            license=_LICENSE,
        )

    def _parse_shot(self, data: dict) -> Shot:
        native_id = str(data.get("shot_id", data.get("id", "")))
        ts_raw = data.get("timestamp") or data.get("date")
        timestamp = None
        if ts_raw:
            try:
                timestamp = datetime.fromisoformat(str(ts_raw))
            except ValueError:
                pass

        ip = data.get("plasma_current") or data.get("ip")
        return Shot(
            shot_id=self.make_shot_id(native_id),
            device_id=self.device_id,
            native_shot_id=native_id,
            timestamp=timestamp,
            duration_s=data.get("duration"),
            plasma_current_MA=float(ip) / 1e6 if ip else None,
            status="disrupted" if data.get("disruption") else "good",
        )

    async def list_diagnostics(self, shot_id: str | None = None) -> DiagnosticList:
        diagnostics = [
            Diagnostic(
                name=canon,
                native_name=f"{source}/{name}",
                canonical_name=canon,
                category=_source_category(source),
                signal_type="scalar",
            )
            for canon, (source, name) in _SIGNAL_MAP.items()
        ]
        return DiagnosticList(
            device_id=self.device_id,
            shot_id=shot_id,
            diagnostics=diagnostics,
            total=len(diagnostics),
        )

    # ------------------------------------------------------------------ #
    # Data retrieval                                                       #
    # ------------------------------------------------------------------ #

    def _resolve_signal(self, diagnostic: str) -> tuple[str, str]:
        if diagnostic in _SIGNAL_MAP:
            return _SIGNAL_MAP[diagnostic]
        # Native format: "source/name"
        if "/" in diagnostic:
            parts = diagnostic.split("/", 1)
            return parts[0], parts[1]
        raise KeyError(f"Unknown diagnostic '{diagnostic}' for MAST")

    async def _fetch_zarr_signal(self, shot_id: str, source: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Fetch a signal from the FAIR-MAST Zarr store via the API."""
        native_id = self.parse_native_id(shot_id)
        client = self._get_client()

        # FAIR-MAST REST endpoint for signal data
        resp = await client.get(f"/shots/{native_id}/sources/{source}/signals/{name}")
        resp.raise_for_status()
        data = resp.json()

        time_arr = np.asarray(data.get("time", data.get("t", [])), dtype=float)
        values = np.asarray(data.get("data", data.get("values", [])), dtype=float)
        return time_arr, values

    async def get_signal(
        self,
        shot_id: str,
        diagnostic: str,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        max_samples: int = 10_000,
    ) -> Signal:
        source, name = self._resolve_signal(diagnostic)
        time_arr, data_arr = await self._fetch_zarr_signal(shot_id, source, name)

        if t_start is not None:
            mask = time_arr >= t_start
            time_arr, data_arr = time_arr[mask], data_arr[mask]
        if t_end is not None:
            mask = time_arr <= t_end
            time_arr, data_arr = time_arr[mask], data_arr[mask]

        return signal_from_arrays(
            shot_id=shot_id,
            diagnostic=diagnostic,
            native_name=f"{source}/{name}",
            time_arr=time_arr,
            data_arr=data_arr,
            max_samples=max_samples,
        )

    async def describe_signal(self, shot_id: str, diagnostic: str) -> SignalSummary:
        source, name = self._resolve_signal(diagnostic)
        time_arr, data_arr = await self._fetch_zarr_signal(shot_id, source, name)
        return summary_from_arrays(
            shot_id=shot_id,
            diagnostic=diagnostic,
            native_name=f"{source}/{name}",
            time_arr=time_arr,
            data_arr=data_arr,
        )

    async def get_equilibrium(self, shot_id: str) -> EquilibriumData | None:
        native_id = self.parse_native_id(shot_id)
        client = self._get_client()

        try:
            resp = await client.get(f"/shots/{native_id}/sources/efm/signals/psi")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.warning("MAST equilibrium unavailable for %s: %s", shot_id, e)
            return None

        time_arr = np.asarray(data.get("time", []), dtype=float)
        psi = np.asarray(data.get("data", []), dtype=float)

        return EquilibriumData(
            shot_id=shot_id,
            reconstruction_code="EFIT",
            time_s=time_arr.tolist(),
            psi_norm=list(np.linspace(0, 1, psi.shape[-1])) if psi.ndim > 1 else psi.tolist(),
            metadata={"license": _LICENSE},
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = await client.get("/shots", params={"limit": 1})
            return resp.status_code == 200
        except Exception as e:
            logger.warning("MAST health check failed: %s", e)
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("MAST HTTP client closed")


def _source_category(source: str) -> str:
    mapping = {
        "amc": "magnetics",
        "efm": "equilibrium",
        "ane": "interferometry",
        "ats": "thomson_scattering",
        "ada": "spectroscopy",
        "bol": "radiation",
        "anb": "heating",
    }
    return mapping.get(source, source)
