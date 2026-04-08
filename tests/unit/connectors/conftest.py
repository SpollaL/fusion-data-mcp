import asyncio

import numpy as np
import pytest

from fusion_data_mcp.connectors.mast import MASTConnector
from fusion_data_mcp.connectors.lhd import LHDConnector


# ------------------------------------------------------------------ #
# MAST fixtures                                                        #
# ------------------------------------------------------------------ #

@pytest.fixture
def mast_connector():
    return MASTConnector()


# ------------------------------------------------------------------ #
# LHD fixtures                                                         #
# ------------------------------------------------------------------ #

@pytest.fixture
def lhd_connector():
    return LHDConnector()


# ------------------------------------------------------------------ #
# Shared fake arrays                                                   #
# ------------------------------------------------------------------ #

@pytest.fixture
def fake_time():
    return np.linspace(0.0, 1.0, 500)


@pytest.fixture
def fake_signal(fake_time):
    return np.sin(2 * np.pi * 10 * fake_time)
