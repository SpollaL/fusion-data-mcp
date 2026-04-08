"""
MAST connector — FAIR-MAST REST API + Zarr object store.

Public API:   https://mastapp.site/json/
Zarr data:    s3://mast/level1/shots/{shot_id}.zarr  (endpoint: https://s3.echo.stfc.ac.uk)
License:      CC BY-SA 4.0
Covers:       MAST campaigns M05–M09

Zarr store layout:
  z[source][signal_name]  → data array
  z[source]['time']       → time axis (seconds)

e.g. z['amc']['plasma_current'], z['amc']['time']
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
_ZARR_ENDPOINT = "https://s3.echo.stfc.ac.uk"
_ZARR_BUCKET = "mast"
_LICENSE = "CC BY-SA 4.0"

# Canonical → (zarr_source_group, zarr_array_name)
# Verified against s3://mast/level1/shots/30420.zarr
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
    """MAST connector via FAIR-MAST public API and Zarr object store."""

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

        try:
            resp = await client.get("/json/shots", params=query_params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error("MAST shot search failed: %s", e)
            return []

        items = data.get("items", []) if isinstance(data, dict) else data
        # The API may not honour limit — enforce it client-side
        return [self._parse_shot(item) for item in items[: params.limit]]

    async def get_shot_metadata(self, shot_id: str) -> ShotMetadata:
        native_id = self.parse_native_id(shot_id)
        client = self._get_client()

        resp = await client.get(f"/json/shots/{native_id}")
        resp.raise_for_status()
        data = resp.json()

        shot = self._parse_shot(data)
        return ShotMetadata(
            **shot.model_dump(),
            additional_fields=to_json_safe({
                k: v for k, v in data.items()
                if k not in ("@context", "@type", "url", "endpoint_url")
            }),
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
        """
        If shot_id given, query the API for all signals on that shot.
        Otherwise return the canonical mapped diagnostics.
        """
        if shot_id is not None:
            native_id = self.parse_native_id(shot_id)
            client = self._get_client()
            try:
                resp = await client.get(f"/json/shots/{native_id}/signals", params={"limit": 200})
                resp.raise_for_status()
                items = resp.json().get("items", [])
                diagnostics = [
                    Diagnostic(
                        name=f"{item['source']}/{item['name']}",
                        native_name=f"{item['source']}/{item['name']}",
                        canonical_name=next(
                            (k for k, (s, n) in _SIGNAL_MAP.items()
                             if s == item["source"] and n == item["name"]),
                            None,
                        ),
                        category=_source_category(item.get("source", "")),
                        description=item.get("description"),
                        signal_type="scalar" if item.get("rank", 1) == 1 else "profile",
                        units=item.get("units"),
                    )
                    for item in items
                ]
                return DiagnosticList(
                    device_id=self.device_id,
                    shot_id=shot_id,
                    diagnostics=diagnostics,
                    total=len(diagnostics),
                )
            except httpx.HTTPError as e:
                logger.warning("Failed to list MAST diagnostics for %s: %s", shot_id, e)

        # Fallback: return canonical mapped diagnostics
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
        if "/" in diagnostic:
            source, name = diagnostic.split("/", 1)
            return source, name
        raise KeyError(f"Unknown diagnostic '{diagnostic}' for MAST")

    async def _read_zarr_signal(
        self, native_id: str, source: str, name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Open the shot's Zarr store and read z[source][name] + z[source]['time'].
        Runs in a thread because zarr 3.x sync reads block.
        """
        def _blocking_read(native_id, source, name):
            import s3fs
            import zarr

            fs = s3fs.S3FileSystem(
                anon=True,
                endpoint_url=_ZARR_ENDPOINT,
                client_kwargs={"region_name": "us-east-1"},
            )
            store = zarr.storage.FsspecStore(
                fs, path=f"{_ZARR_BUCKET}/level1/shots/{native_id}.zarr"
            )
            z = zarr.open(store, mode="r")
            grp = z[source]
            data_arr = np.asarray(grp[name][:], dtype=float)
            time_arr = np.asarray(grp["time"][:], dtype=float)
            return time_arr, data_arr

        return await asyncio.to_thread(_blocking_read, native_id, source, name)

    async def get_signal(
        self,
        shot_id: str,
        diagnostic: str,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        max_samples: int = 10_000,
    ) -> Signal:
        native_id = self.parse_native_id(shot_id)
        source, name = self._resolve_signal(diagnostic)
        time_arr, data_arr = await self._read_zarr_signal(native_id, source, name)

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
        native_id = self.parse_native_id(shot_id)
        source, name = self._resolve_signal(diagnostic)
        time_arr, data_arr = await self._read_zarr_signal(native_id, source, name)
        return summary_from_arrays(
            shot_id=shot_id,
            diagnostic=diagnostic,
            native_name=f"{source}/{name}",
            time_arr=time_arr,
            data_arr=data_arr,
        )

    async def get_equilibrium(self, shot_id: str) -> EquilibriumData | None:
        try:
            native_id = self.parse_native_id(shot_id)
            time_arr, psi_arr = await self._read_zarr_signal(native_id, "efm", "psi")
            _, q_arr = await self._read_zarr_signal(native_id, "efm", "q")
        except Exception as e:
            logger.warning("MAST equilibrium unavailable for %s: %s", shot_id, e)
            return None

        return EquilibriumData(
            shot_id=shot_id,
            reconstruction_code="EFIT",
            time_s=time_arr.tolist(),
            psi_norm=list(np.linspace(0, 1, psi_arr.shape[-1]))
            if psi_arr.ndim > 1
            else psi_arr.tolist(),
            q_profile=q_arr.tolist() if q_arr.ndim == 1 else None,
            metadata={"license": _LICENSE},
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = await client.get("/json/shots", params={"limit": 1})
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
