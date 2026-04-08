"""
Unit tests for ConnectorRegistry.

Focus: routing logic, error types, close_all delegation.
"""

import pytest

from fusion_data_mcp.registry import ConnectorRegistry, UnknownMachineError


class TestRegistry:
    def test_get_unknown_raises_with_device_id(self, registry):
        with pytest.raises(UnknownMachineError) as exc_info:
            registry.get("xplasma")
        assert exc_info.value.device_id == "xplasma"

    def test_get_unknown_error_lists_available(self, registry):
        with pytest.raises(UnknownMachineError) as exc_info:
            registry.get("xplasma")
        assert set(exc_info.value.available) == {"cmod", "mast", "lhd"}

    def test_get_for_shot_routes_by_prefix(self, registry):
        connector = registry.get_for_shot("mast:30420")
        assert connector.device_id == "mast"

    def test_get_for_shot_no_colon_raises_value_error(self, registry):
        with pytest.raises(ValueError, match="device:native_id"):
            registry.get_for_shot("mast30420")

    def test_get_for_shot_unknown_prefix_raises_unknown_machine_error(self, registry):
        with pytest.raises(UnknownMachineError):
            registry.get_for_shot("xplasma:99")

    async def test_close_all_calls_close_on_every_connector(self, mocker):
        reg = ConnectorRegistry()
        connectors = []
        for device_id in ("a", "b", "c"):
            c = mocker.MagicMock()
            c.device_id = device_id
            c.close = mocker.AsyncMock()
            reg.register(c)
            connectors.append(c)

        await reg.close_all()

        for c in connectors:
            c.close.assert_awaited_once()
