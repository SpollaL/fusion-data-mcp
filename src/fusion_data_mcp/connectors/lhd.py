"""
LHD connector — NIFS Large Helical Device data on AWS S3.

Public bucket: s3://nifs-lhd  (region: ap-northeast-1)
Access:        Anonymous (no credentials required)
Coverage:      ~25 years of experiments (1998–present)
License:       NIFS Rights and Terms (open research use)

Actual bucket layout (confirmed):
  s3://nifs-lhd/
    {year}_{Nth}/                   e.g. "2025-26th"
      {DiagName}/                   e.g. "Bolometer", "FIR1"
        {start}-{end}/              e.g. "948600-948699"
          {DiagName}-{shot}-1.zip   per-shot data archive

Each zip contains:
  {DiagName}-{shot}-1.shot         text key=value shot metadata
  {DiagName}-{shot}-1/
    {DiagName}-{shot}-1-{ch}.prm   text key=value channel metadata
    {DiagName}-{shot}-1-{ch}.dat   raw ADC binary (ZLIB-compressed INT16)

Data is raw ADC output. Physical-unit calibration is diagnostic-specific
and is not applied here — returned values are in ADC counts (INT16).
Time axis is seconds from the hardware trigger (0-based).
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import zipfile
import zlib
from typing import Iterator

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

# Canonical name → (diagnostic directory, channel number)
# Confirmed against 2025-26th campaign.
# Note: electron_temperature is not mapped — FastThomson is raw CCD counts
# (requires spectral fitting) and dcece is ~1.8 GB per shot (impractical).
_SIGNAL_MAP: dict[str, tuple[str, int]] = {
    "radiated_power":   ("Bolometer",   1),
    "electron_density": ("FIR1",        1),
    "neutral_beam_power": ("NB1arm",    1),
    "plasma_current":   ("Magnetics_A", 1),
}

# Diagnostic used to enumerate available shots (reliable presence across campaigns)
_INDEX_DIAG = "Bolometer"


class LHDConnector(AbstractConnector):
    """LHD connector via anonymous S3 access + NIFS Labcom zip archives."""

    def __init__(self) -> None:
        self._fs = None
        self._campaigns: list[str] | None = None  # sorted newest-first

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_fs(self):
        if self._fs is None:
            try:
                import s3fs
            except ImportError:
                raise RuntimeError(
                    "s3fs is required for the LHD connector. "
                    "Install it with: pip install s3fs"
                )
            self._fs = s3fs.S3FileSystem(
                anon=True, client_kwargs={"region_name": _REGION}
            )
        return self._fs

    def _list_campaigns(self, fs) -> list[str]:
        """Return campaign directory names sorted newest-first."""
        top = fs.ls(f"s3://{_BUCKET}/", detail=False)
        campaigns = []
        for p in top:
            name = p.rstrip("/").split("/")[-1]
            # Campaign dirs start with a 4-digit year, e.g. "2025-26th"
            if len(name) >= 4 and name[:4].isdigit():
                campaigns.append(name)
        # Sort by leading year, then by trailing number
        def _sort_key(c):
            parts = c.replace("-", "_").split("_")
            return int(parts[0]) if parts[0].isdigit() else 0
        campaigns.sort(key=_sort_key, reverse=True)
        return campaigns

    async def _get_campaigns(self) -> list[str]:
        if self._campaigns is None:
            fs = self._get_fs()
            self._campaigns = await asyncio.to_thread(self._list_campaigns, fs)
        return self._campaigns

    def _shot_range_dir(self, shot_no: int) -> str:
        """Return the range directory name that contains shot_no."""
        base = (shot_no // 100) * 100
        return f"{base:06d}-{base + 99:06d}"

    def _zip_path(self, campaign: str, diag: str, shot_no: int, sub: int = 1) -> str:
        range_dir = self._shot_range_dir(shot_no)
        return f"s3://{_BUCKET}/{campaign}/{diag}/{range_dir}/{diag}-{shot_no}-{sub}.zip"

    def _find_campaign_for_shot(self, fs, campaigns: list[str], shot_no: int) -> str | None:
        """Find which campaign contains shot_no by checking the index diagnostic."""
        for campaign in campaigns:
            range_dir = self._shot_range_dir(shot_no)
            path = f"s3://{_BUCKET}/{campaign}/{_INDEX_DIAG}/{range_dir}/"
            try:
                files = fs.ls(path, detail=False)
                for f in files:
                    fname = f.split("/")[-1]
                    # filename: {Diag}-{shot}-{sub}.zip
                    parts = fname.replace(".zip", "").split("-")
                    if len(parts) >= 2 and parts[-2].isdigit():
                        if int(parts[-2]) == shot_no:
                            return campaign
            except Exception:
                continue
        return None

    def _list_shots_in_campaign(
        self, fs, campaign: str, diag: str, limit: int, offset: int
    ) -> list[int]:
        """Return shot numbers from a campaign's diagnostic directory."""
        try:
            ranges = sorted(
                fs.ls(f"s3://{_BUCKET}/{campaign}/{diag}/", detail=False),
                reverse=True,
            )
        except Exception:
            return []

        shots: list[int] = []
        skipped = 0
        for range_path in ranges:
            try:
                files = fs.ls(range_path, detail=False)
            except Exception:
                continue
            # Parse shot numbers from filenames: {Diag}-{shot}-{sub}.zip
            for fpath in sorted(files, reverse=True):
                fname = fpath.split("/")[-1]
                if not fname.endswith(".zip"):
                    continue
                parts = fname.replace(".zip", "").split("-")
                if len(parts) < 2:
                    continue
                try:
                    shot_no = int(parts[-2])
                except ValueError:
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                shots.append(shot_no)
                if len(shots) >= limit:
                    return shots
        return shots

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
                has_equilibrium=False,
                searchable_by_date=False,
                searchable_by_plasma_params=False,
            ),
            data_license=_LICENSE,
            data_url=f"s3://{_BUCKET}",
        )

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    async def search_shots(self, params: ShotSearchParams) -> list[Shot]:
        campaigns = await self._get_campaigns()
        fs = self._get_fs()

        shots: list[Shot] = []
        remaining = params.limit
        offset = params.offset

        for campaign in campaigns:
            if remaining <= 0:
                break
            native_ids = await asyncio.to_thread(
                self._list_shots_in_campaign, fs, campaign, _INDEX_DIAG,
                remaining, offset if not shots else 0,
            )
            offset = max(0, offset - 100)  # rough offset carry-over
            for shot_no in native_ids:
                shots.append(Shot(
                    shot_id=self.make_shot_id(str(shot_no)),
                    device_id=self.device_id,
                    native_shot_id=str(shot_no),
                    status="unknown",
                ))
            remaining -= len(native_ids)

        return shots[: params.limit]

    async def get_shot_metadata(self, shot_id: str) -> ShotMetadata:
        native_id = self.parse_native_id(shot_id)
        shot_no = int(native_id)
        campaigns = await self._get_campaigns()
        fs = self._get_fs()

        campaign = await asyncio.to_thread(
            self._find_campaign_for_shot, fs, campaigns, shot_no
        )

        additional: dict = {"campaign": campaign} if campaign else {}

        if campaign:
            zip_path = self._zip_path(campaign, _INDEX_DIAG, shot_no)
            try:
                meta = await asyncio.to_thread(self._read_shot_meta, fs, zip_path)
                additional.update(meta)
            except Exception as e:
                logger.warning("LHD metadata read failed for %s: %s", shot_id, e)

        return ShotMetadata(
            shot_id=self.make_shot_id(native_id),
            device_id=self.device_id,
            native_shot_id=native_id,
            additional_fields=to_json_safe(additional),
            diagnostic_count=len(_SIGNAL_MAP),
            license=_LICENSE,
        )

    def _read_shot_meta(self, fs, zip_path: str) -> dict:
        with fs.open(zip_path, "rb") as f:
            data = f.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            shot_files = [n for n in z.namelist() if n.endswith(".shot")]
            if not shot_files:
                return {}
            raw = z.read(shot_files[0]).decode("utf-8", errors="replace")
        result = {}
        for line in raw.splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 3:
                result[parts[1]] = parts[2]
        return result

    async def list_diagnostics(self, shot_id: str | None = None) -> DiagnosticList:
        diagnostics = [
            Diagnostic(
                name=canon,
                native_name=f"{diag}/ch{ch}",
                canonical_name=canon,
                category=diag,
                signal_type="scalar",
                description="Raw ADC counts (INT16). Physical calibration not applied.",
            )
            for canon, (diag, ch) in _SIGNAL_MAP.items()
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

    def _resolve_signal(self, diagnostic: str) -> tuple[str, int]:
        if diagnostic in _SIGNAL_MAP:
            return _SIGNAL_MAP[diagnostic]
        # Allow native "DiagName/chN" or just "DiagName"
        if "/" in diagnostic:
            diag, ch_str = diagnostic.split("/", 1)
            ch = int(ch_str.replace("ch", "")) if ch_str.replace("ch", "").isdigit() else 1
            return diag, ch
        return diagnostic, 1

    async def _load_signal_arrays(
        self, shot_id: str, diagnostic: str
    ) -> tuple[np.ndarray, np.ndarray]:
        native_id = self.parse_native_id(shot_id)
        shot_no = int(native_id)
        diag, ch = self._resolve_signal(diagnostic)

        campaigns = await self._get_campaigns()
        fs = self._get_fs()

        campaign = await asyncio.to_thread(
            self._find_campaign_for_shot, fs, campaigns, shot_no
        )
        if campaign is None:
            raise KeyError(f"Shot {shot_no} not found in any LHD campaign")

        zip_path = self._zip_path(campaign, diag, shot_no)

        def _read(fs, path, ch):
            with fs.open(path, "rb") as f:
                raw = f.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                return _parse_channel(z, ch)

        return await asyncio.to_thread(_read, fs, zip_path, ch)

    async def get_signal(
        self,
        shot_id: str,
        diagnostic: str,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        max_samples: int = 10_000,
    ) -> Signal:
        diag, ch = self._resolve_signal(diagnostic)
        time_arr, data_arr = await self._load_signal_arrays(shot_id, diagnostic)

        if t_start is not None:
            mask = time_arr >= t_start
            time_arr, data_arr = time_arr[mask], data_arr[mask]
        if t_end is not None:
            mask = time_arr <= t_end
            time_arr, data_arr = time_arr[mask], data_arr[mask]

        return signal_from_arrays(
            shot_id=shot_id,
            diagnostic=diagnostic,
            native_name=f"{diag}/ch{ch}",
            time_arr=time_arr,
            data_arr=data_arr,
            max_samples=max_samples,
        )

    async def describe_signal(self, shot_id: str, diagnostic: str) -> SignalSummary:
        diag, ch = self._resolve_signal(diagnostic)
        time_arr, data_arr = await self._load_signal_arrays(shot_id, diagnostic)
        return summary_from_arrays(
            shot_id=shot_id,
            diagnostic=diagnostic,
            native_name=f"{diag}/ch{ch}",
            time_arr=time_arr,
            data_arr=data_arr,
        )

    async def get_equilibrium(self, shot_id: str) -> EquilibriumData | None:
        # LHD VMEC equilibria are not yet available in this bucket
        return None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        try:
            campaigns = await self._get_campaigns()
            return len(campaigns) > 0
        except Exception as e:
            logger.warning("LHD health check failed: %s", e)
            return False

    async def close(self) -> None:
        self._fs = None
        self._campaigns = None
        logger.info("LHD S3 filesystem released")


# ------------------------------------------------------------------ #
# Labcom zip parser                                                    #
# ------------------------------------------------------------------ #

def _parse_prm(z: zipfile.ZipFile, ch: int, prefix: str) -> dict:
    """Parse a .prm file and return key→value dict."""
    prm_name = f"{prefix}/{prefix}-{ch}.prm"
    # Try alternative naming
    candidates = [n for n in z.namelist() if n.endswith(f"-{ch}.prm")]
    if not candidates:
        return {}
    raw = z.read(candidates[0]).decode("utf-8", errors="replace")
    result = {}
    for line in raw.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 3:
            result[parts[1]] = parts[2]
    return result


def _parse_channel(z: zipfile.ZipFile, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (time_arr, data_arr) for channel `ch` from a Labcom zip.

    The .dat file contains raw INT16 data, optionally ZLIB-compressed.
    Time axis is reconstructed from the sampling rate in the .prm file.
    """
    # Find the prefix (e.g. "Bolometer-948609-1")
    shot_files = [n for n in z.namelist() if n.endswith(".shot")]
    if not shot_files:
        raise ValueError("No .shot file in archive")
    prefix = shot_files[0].replace(".shot", "")

    prm = _parse_prm(z, ch, prefix)

    # Locate the .dat file
    dat_candidates = [n for n in z.namelist() if n.endswith(f"-{ch}.dat")]
    if not dat_candidates:
        raise KeyError(f"Channel {ch} .dat not found in archive")
    dat_raw = z.read(dat_candidates[0])

    # Decompress if needed
    comp_method = prm.get("CompressionMethod", "").upper()
    if "ZLIB" in comp_method and dat_raw:
        try:
            dat_raw = zlib.decompress(dat_raw)
        except zlib.error:
            pass  # may already be uncompressed

    if not dat_raw:
        raise ValueError(f"Channel {ch} .dat is empty")

    # Decode as INT16 (little-endian)
    data_arr = np.frombuffer(dat_raw, dtype="<i2").astype(float)

    # For offset-binary encoding, shift to signed
    coding = prm.get("BinaryCoding", "").lower()
    bits_str = prm.get("Resolution(bit)", "16")
    try:
        bits = int(bits_str)
    except ValueError:
        bits = 16
    if "offset" in coding:
        data_arr -= 2 ** (bits - 1)

    # Build time axis
    sample_rate = _get_sample_rate(prm)
    n = len(data_arr)
    time_arr = np.arange(n, dtype=float) / sample_rate

    return time_arr, data_arr


def _get_sample_rate(prm: dict) -> float:
    """Extract sample rate (Hz) from .prm metadata."""
    for key in ("SamplingTimebase", "ClockSpeed", "IntClockSpeed"):
        val = prm.get(key)
        if val:
            try:
                rate = float(val)
                skip = int(prm.get("SkipSize", 1))
                return rate / max(skip, 1)
            except (ValueError, ZeroDivisionError):
                continue
    return 1_000_000.0  # default: 1 MHz
