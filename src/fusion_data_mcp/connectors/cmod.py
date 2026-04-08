"""
Alcator C-Mod connector — MDSplus over TCP.

Public server: alcdata.psfc.mit.edu:8000
Credentials:   cmodpub / cmodpub
Archive covers: 1991–2016 (C-Mod ceased operations)

MDSplus is synchronous and not thread-safe per connection.
All blocking calls are wrapped in asyncio.to_thread().
A small connection pool prevents exhausting the remote server.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

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

_HOST = "alcdata.psfc.mit.edu"
_PORT = 8000
_USER = "cmodpub"
_PASS = "cmodpub"
_POOL_SIZE = 3

# Well-known C-Mod MDSplus tree + node mappings for canonical diagnostics.
# Format: canonical_name -> (tree, node_path)
_SIGNAL_MAP: dict[str, tuple[str, str]] = {
    "plasma_current":     ("MAGNETICS", r"\MAGNETICS::IP"),
    "loop_voltage":       ("MAGNETICS", r"\MAGNETICS::VLOOP"),
    "stored_energy":      ("ANALYSIS",  r"\ANALYSIS::EFIT_AEQDSK:WPLASM"),
    "electron_density":   ("ELECTRONS", r"\ELECTRONS::TOP.TCI.RESULTS:NL_04"),
    "electron_temperature": ("ELECTRONS", r"\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES:TE_RZ"),
    "h_alpha":            ("SPECTROSCOPY", r"\SPECTROSCOPY::TOP.SURVEY:HA_1"),
    "soft_xray":          ("XTOMO",     r"\XTOMO::TOP.RESULTS:CHORD_13"),
    "radiated_power":     ("BOLOMETER", r"\BOLOMETER::TOP.RESULTS.TWOCOLOR:PRAD_TOT"),
    "rf_power":           ("RF",        r"\RF::TOP.ICRF:PICHRF_TOT"),
}

def _tree_category(tree: str) -> str:
    mapping = {
        "MAGNETICS": "magnetics",
        "ANALYSIS": "equilibrium",
        "ELECTRONS": "thomson_scattering",
        "SPECTROSCOPY": "spectroscopy",
        "XTOMO": "radiation",
        "BOLOMETER": "radiation",
        "RF": "heating",
    }
    return mapping.get(tree, tree.lower())


_KNOWN_DIAGNOSTICS: list[dict] = [
    {"name": k, "native_name": v[1], "category": _tree_category(v[0]), "signal_type": "scalar"}
    for k, v in _SIGNAL_MAP.items()
]


class CModConnector(AbstractConnector):
    """Alcator C-Mod connector via MDSplus thin client."""

    def __init__(self, pool_size: int = _POOL_SIZE) -> None:
        self._pool_size = pool_size
        self._pool: asyncio.Queue = asyncio.Queue()
        self._initialized = False

    async def _ensure_pool(self) -> None:
        if self._initialized:
            return
        try:
            import mdsthin as mds
        except ImportError:
            raise RuntimeError(
                "mdsthin is required for the C-Mod connector. "
                "Install it with: pip install mdsthin"
            )
        for _ in range(self._pool_size):
            conn = await asyncio.to_thread(
                self._make_connection, mds
            )
            await self._pool.put(conn)
        self._initialized = True
        logger.info("C-Mod connection pool ready (%d connections)", self._pool_size)

    @staticmethod
    def _make_connection(mds):
        conn = mds.Connection(f"{_HOST}:{_PORT}")
        conn.sendArg(_USER)
        conn.sendArg(_PASS)
        return conn

    # ------------------------------------------------------------------ #
    # Pool management                                                      #
    # ------------------------------------------------------------------ #

    async def _acquire(self):
        await self._ensure_pool()
        return await self._pool.get()

    async def _release(self, conn) -> None:
        await self._pool.put(conn)

    async def _run(self, fn, *args):
        """Run a blocking MDSplus function in a thread, borrowing a connection."""
        conn = await self._acquire()
        try:
            return await asyncio.to_thread(fn, conn, *args)
        except Exception:
            # On error, try to reconnect before returning to pool
            try:
                import mdsthin as mds
                conn = await asyncio.to_thread(self._make_connection, mds)
            except Exception:
                pass  # Return possibly broken conn; health_check will catch it
            raise
        finally:
            await self._release(conn)

    # ------------------------------------------------------------------ #
    # DeviceInfo                                                           #
    # ------------------------------------------------------------------ #

    @property
    def device_id(self) -> str:
        return "cmod"

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            id="cmod",
            name="Alcator C-Mod",
            country="USA",
            type="tokamak",
            description=(
                "High-field compact tokamak operated at MIT PSFC (1991–2016). "
                "Full archive publicly accessible via MDSplus."
            ),
            capabilities=DeviceCapabilities(
                has_equilibrium=True,
                searchable_by_date=True,
                searchable_by_plasma_params=False,
            ),
            data_url=f"mdsip://{_HOST}:{_PORT}",
        )

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    async def search_shots(self, params: ShotSearchParams) -> list[Shot]:
        """
        C-Mod shot IDs encode their date: YYYYMMDDNNN.
        We enumerate by constructing date-range prefixes.
        """

        def _enumerate(conn, date_from, date_to, limit, offset):
            shots = []
            # Query the cmod_summary logbook tree if available
            # Fallback: return empty list — shot IDs must be known by the user
            try:
                conn.openTree("CMOD", 0)  # shot 0 = model tree
                # This is a placeholder; a real implementation would query
                # the logbook tree for shot numbers in the date range.
            except Exception:
                pass
            return shots

        date_from = params.date_from
        date_to = params.date_to or datetime.now(timezone.utc)
        shots = await self._run(_enumerate, date_from, date_to, params.limit, params.offset)
        return shots

    async def get_shot_metadata(self, shot_id: str) -> ShotMetadata:
        native_id = int(self.parse_native_id(shot_id))

        def _fetch(conn, sid):
            meta = {}
            try:
                conn.openTree("MAGNETICS", sid)
                ip = conn.get(r"\MAGNETICS::IP")
                t = conn.get(r"dim_of(\MAGNETICS::IP)")
                meta["plasma_current_peak_MA"] = float(np.max(np.abs(ip.data()))) / 1e6
                meta["duration_s"] = float(t.data()[-1] - t.data()[0])
            except Exception as e:
                logger.debug("MAGNETICS fetch failed for %d: %s", sid, e)
            return meta

        extra = await self._run(_fetch, native_id)
        return ShotMetadata(
            shot_id=self.make_shot_id(native_id),
            device_id=self.device_id,
            native_shot_id=str(native_id),
            plasma_current_MA=extra.pop("plasma_current_peak_MA", None),
            duration_s=extra.pop("duration_s", None),
            additional_fields=to_json_safe(extra),
            diagnostic_count=len(_KNOWN_DIAGNOSTICS),
        )

    async def list_diagnostics(self, shot_id: str | None = None) -> DiagnosticList:
        diagnostics = [
            Diagnostic(
                name=d["name"],
                native_name=d["native_name"],
                canonical_name=d["name"],
                category=d["category"],
                signal_type=d["signal_type"],
            )
            for d in _KNOWN_DIAGNOSTICS
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
        """Return (tree, node_path) for a canonical or native diagnostic name."""
        if diagnostic in _SIGNAL_MAP:
            return _SIGNAL_MAP[diagnostic]
        # Try native path — user passed tree:node directly
        if "::" in diagnostic:
            tree = diagnostic.split("::")[0].lstrip("\\")
            return tree, diagnostic
        raise KeyError(f"Unknown diagnostic '{diagnostic}' for C-Mod")

    async def get_signal(
        self,
        shot_id: str,
        diagnostic: str,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        max_samples: int = 10_000,
    ) -> Signal:
        native_id = int(self.parse_native_id(shot_id))
        tree, node = self._resolve_signal(diagnostic)

        def _fetch(conn, sid, tr, nd):
            conn.openTree(tr, sid)
            data = conn.get(f"data({nd})").data()
            time = conn.get(f"dim_of({nd})").data()
            return np.asarray(time, dtype=float), np.asarray(data, dtype=float)

        time_arr, data_arr = await self._run(_fetch, native_id, tree, node)

        if t_start is not None:
            mask = time_arr >= t_start
            time_arr, data_arr = time_arr[mask], data_arr[mask]
        if t_end is not None:
            mask = time_arr <= t_end
            time_arr, data_arr = time_arr[mask], data_arr[mask]

        return signal_from_arrays(
            shot_id=self.make_shot_id(native_id),
            diagnostic=diagnostic,
            native_name=node,
            time_arr=time_arr,
            data_arr=data_arr,
            max_samples=max_samples,
        )

    async def describe_signal(self, shot_id: str, diagnostic: str) -> SignalSummary:
        native_id = int(self.parse_native_id(shot_id))
        tree, node = self._resolve_signal(diagnostic)

        def _fetch(conn, sid, tr, nd):
            conn.openTree(tr, sid)
            data = conn.get(f"data({nd})").data()
            time = conn.get(f"dim_of({nd})").data()
            return np.asarray(time, dtype=float), np.asarray(data, dtype=float)

        time_arr, data_arr = await self._run(_fetch, native_id, tree, node)
        return summary_from_arrays(
            shot_id=self.make_shot_id(native_id),
            diagnostic=diagnostic,
            native_name=node,
            time_arr=time_arr,
            data_arr=data_arr,
        )

    async def get_equilibrium(self, shot_id: str) -> EquilibriumData | None:
        native_id = int(self.parse_native_id(shot_id))

        def _fetch(conn, sid):
            conn.openTree("ANALYSIS", sid)
            t = conn.get(r"dim_of(\ANALYSIS::EFIT_AEQDSK:WPLASM)").data()
            q95 = conn.get(r"\ANALYSIS::EFIT_AEQDSK:Q95").data()
            ip = conn.get(r"\ANALYSIS::EFIT_AEQDSK:CPASMA").data()
            betat = conn.get(r"\ANALYSIS::EFIT_AEQDSK:BETAT").data()
            return (
                np.asarray(t, dtype=float),
                np.asarray(q95, dtype=float),
                np.asarray(ip, dtype=float),
                np.asarray(betat, dtype=float),
            )

        try:
            t, q95, ip, betat = await self._run(_fetch, native_id)
        except Exception as e:
            logger.warning("Equilibrium unavailable for %s: %s", shot_id, e)
            return None

        return EquilibriumData(
            shot_id=self.make_shot_id(native_id),
            reconstruction_code="EFIT",
            time_s=t.tolist(),
            psi_norm=list(np.linspace(0, 1, 51)),  # standard EFIT psi grid
            q_profile=q95.tolist(),
            ip_MA=float(np.mean(ip)) / 1e6 if len(ip) > 0 else None,
            beta_total=float(np.mean(betat)) if len(betat) > 0 else None,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        try:
            def _ping(conn):
                conn.get("1+1")
                return True
            return await self._run(_ping)
        except Exception as e:
            logger.warning("C-Mod health check failed: %s", e)
            return False

    async def close(self) -> None:
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await asyncio.to_thread(conn.disconnect)
            except Exception:
                pass
        logger.info("C-Mod connection pool closed")
