import pytest

from fusion_data_mcp.connectors.cmod import CModConnector
from fusion_data_mcp.connectors.lhd import LHDConnector
from fusion_data_mcp.connectors.mast import MASTConnector


@pytest.fixture
async def cmod_connector_live():
    connector = CModConnector(pool_size=1)
    yield connector
    await connector.close()


@pytest.fixture
async def mast_connector_live():
    connector = MASTConnector()
    yield connector
    await connector.close()


@pytest.fixture
async def lhd_connector_live():
    connector = LHDConnector()
    yield connector
    await connector.close()
