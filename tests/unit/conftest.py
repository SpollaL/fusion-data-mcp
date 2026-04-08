import numpy as np
import pytest

from fusion_data_mcp.connectors.base import AbstractConnector
from fusion_data_mcp.models import DeviceCapabilities, DeviceInfo
from fusion_data_mcp.registry import ConnectorRegistry


@pytest.fixture
def uniform_time():
    """1000-point uniform time axis 0..1 s."""
    return np.linspace(0.0, 1.0, 1000)


@pytest.fixture
def clean_data(uniform_time):
    """Sine wave, no NaN/Inf."""
    return np.sin(2 * np.pi * 5 * uniform_time)


@pytest.fixture
def dirty_data(uniform_time):
    """Sine wave with NaN at idx 10, +Inf at idx 50, -Inf at idx 100."""
    d = np.sin(2 * np.pi * 5 * uniform_time).copy()
    d[10] = np.nan
    d[50] = np.inf
    d[100] = -np.inf
    return d


@pytest.fixture
def mock_connector(mocker):
    """A MagicMock spec'd against AbstractConnector with device_id='mock'."""
    conn = mocker.MagicMock(spec=AbstractConnector)
    conn.device_id = "mock"
    conn.device_info = DeviceInfo(
        id="mock",
        name="Mock Device",
        country="XX",
        type="other",
        description="Test",
        capabilities=DeviceCapabilities(
            has_equilibrium=False,
            searchable_by_date=False,
            searchable_by_plasma_params=False,
        ),
    )
    conn.close = mocker.AsyncMock()
    return conn


@pytest.fixture
def registry(mock_connector):
    """Registry pre-populated with connectors for cmod, mast, lhd."""
    reg = ConnectorRegistry()
    for device_id in ("cmod", "mast", "lhd"):
        c = type("C", (), {"device_id": device_id, "close": mock_connector.close})()
        reg.register(c)
    return reg
