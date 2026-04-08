"""
Unit tests for MASTConnector.

Focus: signal resolution, shot parsing edge cases, Zarr read path.
HTTP calls are intercepted with respx; Zarr reads are patched.
"""

import numpy as np
import pytest
import respx
import httpx

from fusion_data_mcp.connectors.mast import MASTConnector, _SIGNAL_MAP
from fusion_data_mcp.models import Signal, SignalSummary, ShotSearchParams


# ------------------------------------------------------------------ #
# _resolve_signal                                                      #
# ------------------------------------------------------------------ #

class TestResolveSignal:
    def test_canonical_name(self, mast_connector):
        source, name = mast_connector._resolve_signal("plasma_current")
        assert source == "amc"
        assert name == "plasma_current"

    def test_native_slash_format(self, mast_connector):
        source, name = mast_connector._resolve_signal("bol/power_radiated_total")
        assert source == "bol"
        assert name == "power_radiated_total"

    def test_unknown_raises_key_error(self, mast_connector):
        with pytest.raises(KeyError):
            mast_connector._resolve_signal("nonsense_signal")


# ------------------------------------------------------------------ #
# _parse_shot                                                          #
# ------------------------------------------------------------------ #

class TestParseShot:
    def test_canonical_shot_id_format(self, mast_connector):
        shot = mast_connector._parse_shot({"shot_id": 30420})
        assert shot.shot_id == "mast:30420"
        assert shot.native_shot_id == "30420"

    def test_null_plasma_current_is_none(self, mast_connector):
        shot = mast_connector._parse_shot({"shot_id": 1, "plasma_current": None})
        assert shot.plasma_current_MA is None

    def test_plasma_current_converted_to_ma(self, mast_connector):
        shot = mast_connector._parse_shot({"shot_id": 1, "plasma_current": 400_000})
        assert shot.plasma_current_MA == pytest.approx(0.4)

    def test_disruption_flag_sets_status(self, mast_connector):
        shot = mast_connector._parse_shot({"shot_id": 1, "disruption": True})
        assert shot.status == "disrupted"

    def test_no_disruption_is_good(self, mast_connector):
        shot = mast_connector._parse_shot({"shot_id": 1, "disruption": False})
        assert shot.status == "good"

    def test_missing_timestamp_is_none(self, mast_connector):
        shot = mast_connector._parse_shot({"shot_id": 1})
        assert shot.timestamp is None

    def test_alternate_id_key(self, mast_connector):
        """API sometimes uses 'id' instead of 'shot_id'."""
        shot = mast_connector._parse_shot({"id": 999})
        assert shot.native_shot_id == "999"


# ------------------------------------------------------------------ #
# search_shots — HTTP mocked with respx                                #
# ------------------------------------------------------------------ #

class TestSearchShots:
    @respx.mock
    async def test_uses_json_prefix_endpoint(self, mast_connector):
        route = respx.get("https://mastapp.site/json/shots").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        params = ShotSearchParams(device_id="mast", limit=5)
        await mast_connector.search_shots(params)
        assert route.called

    @respx.mock
    async def test_limit_enforced_client_side(self, mast_connector):
        items = [{"shot_id": i} for i in range(50)]
        respx.get("https://mastapp.site/json/shots").mock(
            return_value=httpx.Response(200, json={"items": items})
        )
        params = ShotSearchParams(device_id="mast", limit=3)
        shots = await mast_connector.search_shots(params)
        assert len(shots) == 3

    @respx.mock
    async def test_http_error_returns_empty_list(self, mast_connector):
        respx.get("https://mastapp.site/json/shots").mock(
            return_value=httpx.Response(503)
        )
        params = ShotSearchParams(device_id="mast", limit=5)
        shots = await mast_connector.search_shots(params)
        assert shots == []


# ------------------------------------------------------------------ #
# get_signal — Zarr read patched                                       #
# ------------------------------------------------------------------ #

class TestGetSignal:
    async def test_returns_signal_model(self, mast_connector, mocker, fake_time, fake_signal):
        mocker.patch.object(
            mast_connector,
            "_read_zarr_signal",
            return_value=(fake_time, fake_signal),
        )
        sig = await mast_connector.get_signal("mast:30420", "plasma_current")
        assert isinstance(sig, Signal)
        assert sig.shot_id == "mast:30420"
        assert sig.diagnostic == "plasma_current"

    async def test_time_filtering_applied(self, mast_connector, mocker, fake_time, fake_signal):
        mocker.patch.object(
            mast_connector,
            "_read_zarr_signal",
            return_value=(fake_time, fake_signal),
        )
        sig = await mast_connector.get_signal(
            "mast:30420", "plasma_current", t_start=0.3, t_end=0.7
        )
        assert min(sig.time_s) >= 0.3 - 1e-9
        assert max(sig.time_s) <= 0.7 + 1e-9

    async def test_native_name_in_output(self, mast_connector, mocker, fake_time, fake_signal):
        mocker.patch.object(
            mast_connector,
            "_read_zarr_signal",
            return_value=(fake_time, fake_signal),
        )
        sig = await mast_connector.get_signal("mast:30420", "plasma_current")
        assert sig.native_name == "amc/plasma_current"


# ------------------------------------------------------------------ #
# health_check                                                         #
# ------------------------------------------------------------------ #

class TestHealthCheck:
    @respx.mock
    async def test_200_returns_true(self, mast_connector):
        respx.get("https://mastapp.site/json/shots").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        assert await mast_connector.health_check() is True

    @respx.mock
    async def test_non_200_returns_false(self, mast_connector):
        respx.get("https://mastapp.site/json/shots").mock(
            return_value=httpx.Response(503)
        )
        assert await mast_connector.health_check() is False

    @respx.mock
    async def test_network_error_returns_false(self, mast_connector):
        respx.get("https://mastapp.site/json/shots").mock(
            side_effect=httpx.ConnectError("unreachable")
        )
        assert await mast_connector.health_check() is False
