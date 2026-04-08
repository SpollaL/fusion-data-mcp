"""
LHD connector — NIFS Large Helical Device data on AWS S3.

Public bucket: s3://nifs-lhd  (region: ap-northeast-1)
Access:        Anonymous (no credentials required)
Coverage:      ~25 years of experiments (1998–present), ~2 PB
License:       NIFS Rights and Terms (open research use)

Files are HDF5/NetCDF accessed via s3fs + xarray.
Bucket layout is discovered at init time.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Literal

import numpy as np

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

_BUCKET = "nifs-lhd"
_REGION = "ap-northeast-1"
_LICENSE = "NIFS Rights and Terms — https://www-lhd.nifs.ac.jp/pub/RightsTerms.html"

BucketLayout = Literal["partitioned", "flat", "manifest", "unknown"]

# Known canonical → LHD native group/variable mappings (best-effort).
# These will be refined once the actual bucket layout is confirmed.
_SIGNAL_MAP: dict[str, tuple[str, str]] = {
    "plasma_current":       ("magnetics", "ip"),
    "loop_voltage":         ("magnetics", "vloop"),
    "stored_energy":        ("mhd", "wmhd"),
    "electron_density":     ("thomson", "ne"),
    "electron_temperature": ("thomson", "te"),
    "neutral_beam_power":   ("heating", "pnbi"),
    "radiated_power":       ("bolometer", "prad"),
}


class LHDConnector(AbstractConnector):
    """LHD connector via anonymous S3 access + HDF5/NetCDF."""

    def __init__(self) -> None:
        self._fs = None
        self._layout: BucketLayout = "unknown"

    def _get_fs(self):
        if self._fs is None:
            try:
                import s3fs
            except ImportError:
                raise RuntimeError(
                    "s3fs is required for the LHD connector. "
                    "Install it with: pip install s3fs"
                )
            self._fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": _REGION})
        return self._fs

    async def _discover_layout(self) -> BucketLayout:
        """Probe the bucket top-level structure to infer layout convention."""
        if self._layout != "unknown":
            return self._layout

        def _probe(fs):
            try:
                top = fs.ls(f"s3://{_BUCKET}/", detail=False)
                if not top:
                    return "unknown"
                # Check for a manifest/index file
                manifest_candidates = [p for p in top if "index" in p.lower() or "manifest" in p.lower()]
                if manifest_candidates:
                    return "manifest"
                # Check if top-level entries look like date partitions (YYYY or YYYYMM)
                names = [p.rstrip("/").split("/")[-1] for p in top[:10]]
                if all(n.isdigit() and len(n) in (4, 6, 8) for n in names if n):
                    return "partitioned"
                return "flat"
            except Exception as e:
                logger.warning("LHD bucket probe failed: %s", e)
                return "unknown"

        layout = await asyncio.to_thread(_probe, self._get_fs())
        self._layout = layout
        logger.info("LHD bucket layout detected: %s", layout)
        return layout

    def _shot_path(self, native_id: str) -> str:
        """Construct the S3 path for a given shot ID."""
        # Partitioned layout assumption: shots/YYYY/YYYYMMDD/SHOTID.h5
        # This will need adjustment based on actual bucket structure.
        year = native_id[:4] if len(native_id) >= 4 else native_id
        return f"s3://{_BUCKET}/shots/{year}/{native_id}.h5"

    # ------------------------------------------------------------------ #
    # DeviceInfo                                                           #
    # ------------------------------------------------------------------ #

    @property
    def device_id(self) -> str:
        return "lhd"

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            id="lhd",
            name="Large Helical Device",
            country="Japan",
            type="stellarator",
            description=(
                "World's largest superconducting stellarator at NIFS, Japan. "
                "~2 PB of experimental data (1998–present) publicly available on AWS S3."
            ),
            capabilities=DeviceCapabilities(
                has_equilibrium=True,
                searchable_by_date=True,
                searchable_by_plasma_params=False,
                max_signal_duration_s=10.0,
            ),
            data_license=_LICENSE,
            data_url=f"s3://{_BUCKET}",
        )

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    async def search_shots(self, params: ShotSearchParams) -> list[Shot]:
        await self._discover_layout()
        fs = self._get_fs()

        def _list(fs, date_from, date_to, limit, offset):
            shots = []
            try:
                if self._layout == "partitioned":
                    # List year prefixes within date range
                    year_from = date_from.year if date_from else 1998
                    year_to = date_to.year if date_to else datetime.now().year
                    for year in range(year_from, year_to + 1):
                        prefix = f"s3://{_BUCKET}/shots/{year}/"
                        try:
                            paths = fs.ls(prefix, detail=False)
                            for p in paths[offset : offset + limit - len(shots)]:
                                native_id = p.rstrip("/").split("/")[-1].replace(".h5", "")
                                shots.append(native_id)
                                if len(shots) >= limit:
                                    return shots
                        except Exception:
                            continue
                else:
                    # Flat layout: list top-level files
                    paths = fs.ls(f"s3://{_BUCKET}/", detail=False)
                    for p in paths[offset : offset + limit]:
                        native_id = p.rstrip("/").split("/")[-1].replace(".h5", "")
                        shots.append(native_id)
            except Exception as e:
                logger.error("LHD shot listing failed: %s", e)
            return shots

        native_ids = await asyncio.to_thread(
            _list, fs, params.date_from, params.date_to, params.limit, params.offset
        )
        return [
            Shot(
                shot_id=self.make_shot_id(nid),
                device_id=self.device_id,
                native_shot_id=nid,
                status="unknown",
            )
            for nid in native_ids
        ]

    async def get_shot_metadata(self, shot_id: str) -> ShotMetadata:
        native_id = self.parse_native_id(shot_id)
        path = self._shot_path(native_id)
        fs = self._get_fs()

        def _read_attrs(fs, p):
            try:
                import h5py
                with fs.open(p, "rb") as f:
                    with h5py.File(f, "r") as h:
                        attrs = dict(h.attrs)
                        groups = list(h.keys())
                return attrs, groups
            except Exception as e:
                logger.warning("LHD metadata read failed for %s: %s", p, e)
                return {}, []

        attrs, groups = await asyncio.to_thread(_read_attrs, fs, path)
        return ShotMetadata(
            shot_id=self.make_shot_id(native_id),
            device_id=self.device_id,
            native_shot_id=native_id,
            additional_fields=to_json_safe({**attrs, "groups": groups}),
            diagnostic_count=len(groups),
            license=_LICENSE,
        )

    async def list_diagnostics(self, shot_id: str | None = None) -> DiagnosticList:
        if shot_id is not None:
            native_id = self.parse_native_id(shot_id)
            path = self._shot_path(native_id)
            fs = self._get_fs()

            def _list_groups(fs, p):
                try:
                    import h5py
                    with fs.open(p, "rb") as f:
                        with h5py.File(f, "r") as h:
                            return list(h.keys())
                except Exception:
                    return []

            groups = await asyncio.to_thread(_list_groups, fs, path)
            diagnostics = [
                Diagnostic(
                    name=g,
                    native_name=g,
                    canonical_name=next(
                        (k for k, (grp, _) in _SIGNAL_MAP.items() if grp == g), None
                    ),
                    category=g,
                    signal_type="unknown",
                )
                for g in groups
            ]
        else:
            # Return known canonical diagnostics
            diagnostics = [
                Diagnostic(
                    name=canon,
                    native_name=f"{grp}/{var}",
                    canonical_name=canon,
                    category=grp,
                    signal_type="scalar",
                )
                for canon, (grp, var) in _SIGNAL_MAP.items()
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
            parts = diagnostic.split("/", 1)
            return parts[0], parts[1]
        raise KeyError(f"Unknown diagnostic '{diagnostic}' for LHD")

    async def _load_signal_arrays(
        self, native_id: str, group: str, variable: str
    ) -> tuple[np.ndarray, np.ndarray]:
        path = self._shot_path(native_id)
        fs = self._get_fs()

        def _read(fs, p, grp, var):
            import h5py
            with fs.open(p, "rb") as f:
                with h5py.File(f, "r") as h:
                    ds = h[grp][var]
                    data = ds[:]
                    # Time dimension: look for 'time' or 't' coordinate
                    time = None
                    for tname in ("time", "t", "TIME"):
                        if tname in h[grp]:
                            time = h[grp][tname][:]
                            break
                    if time is None and "time" in ds.dims:
                        time = ds.dims[0][:]
                    if time is None:
                        time = np.arange(len(data), dtype=float)
                    return np.asarray(time, dtype=float), np.asarray(data, dtype=float)

        return await asyncio.to_thread(_read, fs, path, group, variable)

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
        group, variable = self._resolve_signal(diagnostic)
        time_arr, data_arr = await self._load_signal_arrays(native_id, group, variable)

        if t_start is not None:
            mask = time_arr >= t_start
            time_arr, data_arr = time_arr[mask], data_arr[mask]
        if t_end is not None:
            mask = time_arr <= t_end
            time_arr, data_arr = time_arr[mask], data_arr[mask]

        return signal_from_arrays(
            shot_id=self.make_shot_id(native_id),
            diagnostic=diagnostic,
            native_name=f"{group}/{variable}",
            time_arr=time_arr,
            data_arr=data_arr,
            max_samples=max_samples,
        )

    async def describe_signal(self, shot_id: str, diagnostic: str) -> SignalSummary:
        native_id = self.parse_native_id(shot_id)
        group, variable = self._resolve_signal(diagnostic)
        time_arr, data_arr = await self._load_signal_arrays(native_id, group, variable)
        return summary_from_arrays(
            shot_id=self.make_shot_id(native_id),
            diagnostic=diagnostic,
            native_name=f"{group}/{variable}",
            time_arr=time_arr,
            data_arr=data_arr,
        )

    async def get_equilibrium(self, shot_id: str) -> EquilibriumData | None:
        native_id = self.parse_native_id(shot_id)
        # LHD uses VMEC for equilibrium reconstruction
        path = self._shot_path(native_id)
        fs = self._get_fs()

        def _read_vmec(fs, p):
            import h5py
            with fs.open(p, "rb") as f:
                with h5py.File(f, "r") as h:
                    if "vmec" not in h and "VMEC" not in h:
                        return None
                    grp = h.get("vmec") or h.get("VMEC")
                    psi = grp["psi_norm"][:] if "psi_norm" in grp else np.linspace(0, 1, 51)
                    q = grp["q"][:] if "q" in grp else None
                    return psi, q

        try:
            result = await asyncio.to_thread(_read_vmec, fs, path)
        except Exception as e:
            logger.warning("LHD equilibrium unavailable for %s: %s", shot_id, e)
            return None

        if result is None:
            return None

        psi, q = result
        return EquilibriumData(
            shot_id=self.make_shot_id(native_id),
            reconstruction_code="VMEC",
            time_s=[0.0],  # VMEC is typically a single-time reconstruction
            psi_norm=psi.tolist(),
            q_profile=q.tolist() if q is not None else None,
            metadata={"license": _LICENSE},
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        try:
            fs = self._get_fs()
            result = await asyncio.to_thread(fs.ls, f"s3://{_BUCKET}/", detail=False)
            return len(result) > 0
        except Exception as e:
            logger.warning("LHD health check failed: %s", e)
            return False

    async def close(self) -> None:
        self._fs = None
        logger.info("LHD S3 filesystem released")
