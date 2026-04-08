"""
Integration tests for the MAST connector (FAIR-MAST public API).

Requires live access to mastapp.site (no auth, CC BY-SA 4.0).
Run with: pytest --run-integration tests/integration/test_mast_live.py -v
"""

import json

import pytest

from fusion_data_mcp.models import Shot, Signal, SignalSummary, ShotSearchParams

# MAST shot from campaign M9, well-documented in literature
KNOWN_SHOT = "mast:30420"


@pytest.mark.integration
async def test_mast_health_check(mast_connector_live):
    ok = await mast_connector_live.health_check()
    assert ok is True, "FAIR-MAST API is unreachable — check mastapp.site"


@pytest.mark.integration
async def test_mast_device_info(mast_connector_live):
    info = mast_connector_live.device_info
    assert info.id == "mast"
    assert info.type == "tokamak"
    assert info.country == "UK"
    assert info.data_license == "CC BY-SA 4.0"
    assert info.capabilities.searchable_by_plasma_params is True


@pytest.mark.integration
async def test_mast_search_shots_returns_results(mast_connector_live):
    params = ShotSearchParams(device_id="mast", limit=5)
    shots = await mast_connector_live.search_shots(params)

    assert len(shots) > 0
    for shot in shots:
        assert isinstance(shot, Shot)
        assert shot.device_id == "mast"
        assert shot.shot_id.startswith("mast:")
        assert shot.native_shot_id != ""


@pytest.mark.integration
async def test_mast_search_shots_respects_limit(mast_connector_live):
    params = ShotSearchParams(device_id="mast", limit=3)
    shots = await mast_connector_live.search_shots(params)
    assert len(shots) <= 3


@pytest.mark.integration
async def test_mast_get_shot_metadata(mast_connector_live):
    meta = await mast_connector_live.get_shot_metadata(KNOWN_SHOT)

    assert meta.shot_id == KNOWN_SHOT
    assert meta.device_id == "mast"
    assert meta.native_shot_id == "30420"
    assert meta.license == "CC BY-SA 4.0"


@pytest.mark.integration
async def test_mast_list_diagnostics(mast_connector_live):
    result = await mast_connector_live.list_diagnostics()
    assert result.device_id == "mast"
    assert result.total > 0
    names = [d.name for d in result.diagnostics]
    assert "plasma_current" in names


@pytest.mark.integration
async def test_mast_get_signal_plasma_current(mast_connector_live):
    signal = await mast_connector_live.get_signal(KNOWN_SHOT, "plasma_current")

    assert isinstance(signal, Signal)
    assert signal.shot_id == KNOWN_SHOT
    assert signal.diagnostic == "plasma_current"
    assert len(signal.time_s) > 0
    assert len(signal.data) == len(signal.time_s)


@pytest.mark.integration
async def test_mast_get_signal_time_filtering(mast_connector_live):
    signal = await mast_connector_live.get_signal(
        KNOWN_SHOT, "plasma_current", t_start=0.1, t_end=0.3
    )
    if len(signal.time_s) > 0:
        assert min(signal.time_s) >= 0.1 - 1e-9
        assert max(signal.time_s) <= 0.3 + 1e-9


@pytest.mark.integration
async def test_mast_describe_signal(mast_connector_live):
    summary = await mast_connector_live.describe_signal(KNOWN_SHOT, "plasma_current")

    assert isinstance(summary, SignalSummary)
    assert summary.n_samples > 0
    assert summary.duration_s > 0


@pytest.mark.integration
async def test_mast_get_equilibrium(mast_connector_live):
    eq = await mast_connector_live.get_equilibrium(KNOWN_SHOT)
    # MAST EFIT may or may not be available via the public API endpoint
    if eq is not None:
        assert eq.shot_id == KNOWN_SHOT
        assert eq.reconstruction_code == "EFIT"
        assert len(eq.psi_norm) > 0


@pytest.mark.integration
async def test_mast_signal_is_json_serializable(mast_connector_live):
    signal = await mast_connector_live.get_signal(KNOWN_SHOT, "plasma_current", max_samples=200)
    json.dumps(signal.model_dump())


@pytest.mark.integration
async def test_mast_native_diagnostic_name(mast_connector_live):
    """Verify native 'source/name' format also resolves correctly."""
    signal = await mast_connector_live.get_signal(KNOWN_SHOT, "amc/plasma_current")
    assert isinstance(signal, Signal)
    assert len(signal.time_s) > 0
