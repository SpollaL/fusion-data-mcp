"""ConnectorRegistry: routes requests to the correct data source connector."""

from __future__ import annotations

from .connectors.base import AbstractConnector


class UnknownMachineError(Exception):
    def __init__(self, device_id: str, available: list[str]) -> None:
        self.device_id = device_id
        self.available = available
        super().__init__(
            f"Unknown machine '{device_id}'. Available: {available}"
        )


class ConnectorRegistry:
    """
    Singleton registry of all active connectors.
    Instantiated once at server startup; connectors are singletons within it.
    """

    def __init__(self) -> None:
        self._connectors: dict[str, AbstractConnector] = {}

    def register(self, connector: AbstractConnector) -> None:
        self._connectors[connector.device_id] = connector

    def get(self, device_id: str) -> AbstractConnector:
        """Return the connector for the given machine ID."""
        if device_id not in self._connectors:
            raise UnknownMachineError(device_id, list(self._connectors.keys()))
        return self._connectors[device_id]

    def get_for_shot(self, shot_id: str) -> AbstractConnector:
        """
        Route by parsing the device prefix from a canonical shot_id.
        e.g. 'cmod:1120815012' → CModConnector
        """
        if ":" not in shot_id:
            raise ValueError(
                f"Cannot route shot_id '{shot_id}': expected format 'device:native_id'"
            )
        device_id = shot_id.split(":", 1)[0]
        return self.get(device_id)

    def list_all(self) -> list[AbstractConnector]:
        return list(self._connectors.values())

    async def close_all(self) -> None:
        for connector in self._connectors.values():
            await connector.close()
