"""
Integration tests for the Alcator C-Mod connector.

Requires live access to alcdata.psfc.mit.edu:8000 (public, cmodpub/cmodpub).
Run with: pytest --run-integration tests/integration/test_cmod_live.py -v

NOTE: Port 8000 may be blocked by firewalls on some networks. Tests that need
a live connection are automatically skipped if the port is unreachable.
Tests of static properties (device_info, list_diagnostics) always run.
"""

import json
import socket

import pytest

from fusion_data_mcp.models import EquilibriumData, Signal, SignalSummary


def _cmod_port_reachable() -> bool:
    try:
        with socket.create_connection(("alcdata.psfc.mit.edu", 8000), timeout=5):
            return True
    except OSError:
        return False


cmod_reachable = _cmod_port_reachable()
skip_if_cmod_unreachable = pytest.mark.skipif(
    not cmod_reachable,
    reason="alcdata.psfc.mit.edu:8000 unreachable (port blocked or server down)",
)

# A well-documented C-Mod shot used in many published analyses
KNOWN_SHOT = "cmod:1120815012"
KNOWN_NATIVE_ID = 1120815012


# ------------------------------------------------------------------ #
# Static tests — no network connection required                        #
# ------------------------------------------------------------------ #

@pytest.mark.integration
async def test_cmod_device_info(cmod_connector_live):
    info = cmod_connector_live.device_info
    assert info.id == "cmod"
    assert info.type == "tokamak"
    assert info.country == "USA"
    assert info.capabilities.has_equilibrium is True


@pytest.mark.integration
async def test_cmod_list_diagnostics(cmod_connector_live):
    result = await cmod_connector_live.list_diagnostics()
    assert result.device_id == "cmod"
    assert result.total > 0
    names = [d.name for d in result.diagnostics]
    assert "plasma_current" in names
    assert "electron_density" in names


# ------------------------------------------------------------------ #
# Live connection tests                                                 #
# ------------------------------------------------------------------ #

@pytest.mark.integration
@skip_if_cmod_unreachable
async def test_cmod_health_check(cmod_connector_live):
    ok = await cmod_connector_live.health_check()
    assert ok is True, "C-Mod MDSplus server unreachable — check alcdata.psfc.mit.edu:8000"


@pytest.mark.integration
@skip_if_cmod_unreachable
async def test_cmod_get_shot_metadata(cmod_connector_live):
    meta = await cmod_connector_live.get_shot_metadata(KNOWN_SHOT)
    assert meta.shot_id == KNOWN_SHOT
    assert meta.device_id == "cmod"
    assert meta.native_shot_id == str(KNOWN_NATIVE_ID)


@pytest.mark.integration
@skip_if_cmod_unreachable
async def test_cmod_get_signal_plasma_current(cmod_connector_live):
    signal = await cmod_connector_live.get_signal(KNOWN_SHOT, "plasma_current")

    assert isinstance(signal, Signal)
    assert signal.shot_id == KNOWN_SHOT
    assert signal.diagnostic == "plasma_current"
    assert len(signal.time_s) > 0
    assert len(signal.data) == len(signal.time_s)
    # C-Mod plasma currents are typically 0.5–2 MA; check order of magnitude
    max_current = max(abs(v) for v in signal.data if v is not None)
    assert 1e5 < max_current < 3e6, f"Unexpected plasma current range: {max_current}"


@pytest.mark.integration
@skip_if_cmod_unreachable
async def test_cmod_get_signal_time_filtering(cmod_connector_live):
    signal = await cmod_connector_live.get_signal(
        KNOWN_SHOT, "plasma_current", t_start=0.5, t_end=1.0
    )
    assert len(signal.time_s) > 0
    assert min(signal.time_s) >= 0.5 - 1e-9
    assert max(signal.time_s) <= 1.0 + 1e-9


@pytest.mark.integration
@skip_if_cmod_unreachable
async def test_cmod_get_signal_downsampling(cmod_connector_live):
    signal = await cmod_connector_live.get_signal(
        KNOWN_SHOT, "plasma_current", max_samples=100
    )
    assert len(signal.time_s) <= 100
    if signal.downsampled:
        assert signal.original_n_samples is not None
        assert signal.original_n_samples > 100


@pytest.mark.integration
@skip_if_cmod_unreachable
async def test_cmod_describe_signal(cmod_connector_live):
    summary = await cmod_connector_live.describe_signal(KNOWN_SHOT, "plasma_current")

    assert isinstance(summary, SignalSummary)
    assert summary.shot_id == KNOWN_SHOT
    assert summary.n_samples > 0
    assert summary.duration_s > 0
    assert summary.mean is not None
    assert summary.min is not None
    assert summary.max is not None
    assert summary.max >= summary.min


@pytest.mark.integration
@skip_if_cmod_unreachable
async def test_cmod_get_equilibrium(cmod_connector_live):
    eq = await cmod_connector_live.get_equilibrium(KNOWN_SHOT)
    if eq is not None:
        assert isinstance(eq, EquilibriumData)
        assert eq.shot_id == KNOWN_SHOT
        assert eq.reconstruction_code == "EFIT"
        assert len(eq.time_s) > 0
        assert len(eq.psi_norm) > 0


@pytest.mark.integration
@skip_if_cmod_unreachable
async def test_cmod_signal_output_is_json_serializable(cmod_connector_live):
    signal = await cmod_connector_live.get_signal(KNOWN_SHOT, "plasma_current", max_samples=200)
    json.dumps(signal.model_dump())
