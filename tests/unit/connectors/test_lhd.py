"""
Unit tests for LHDConnector.

Focus: path construction, layout detection, signal resolution.
S3 and h5py calls are patched — no real network.
"""

import pytest

from fusion_data_mcp.connectors.lhd import LHDConnector


class TestShotPath:
    def test_year_extracted_from_native_id(self, lhd_connector):
        path = lhd_connector._shot_path("184255")
        assert "1842" in path  # first 4 chars
        assert "184255" in path

    def test_full_path_contains_bucket(self, lhd_connector):
        path = lhd_connector._shot_path("184255")
        assert "nifs-lhd" in path

    def test_short_id_does_not_crash(self, lhd_connector):
        # Should not raise IndexError for very short IDs
        path = lhd_connector._shot_path("99")
        assert "99" in path


class TestDiscoverLayout:
    async def test_detects_partitioned(self, lhd_connector, mocker):
        mocker.patch.object(
            lhd_connector,
            "_get_fs",
            return_value=mocker.MagicMock(
                ls=mocker.MagicMock(return_value=["nifs-lhd/1998", "nifs-lhd/1999", "nifs-lhd/2000"])
            ),
        )
        layout = await lhd_connector._discover_layout()
        assert layout == "partitioned"

    async def test_detects_manifest(self, lhd_connector, mocker):
        mocker.patch.object(
            lhd_connector,
            "_get_fs",
            return_value=mocker.MagicMock(
                ls=mocker.MagicMock(return_value=["nifs-lhd/manifest.json", "nifs-lhd/index.csv"])
            ),
        )
        layout = await lhd_connector._discover_layout()
        assert layout == "manifest"

    async def test_probe_failure_returns_unknown(self, lhd_connector, mocker):
        mocker.patch.object(
            lhd_connector,
            "_get_fs",
            return_value=mocker.MagicMock(
                ls=mocker.MagicMock(side_effect=PermissionError("blocked"))
            ),
        )
        layout = await lhd_connector._discover_layout()
        assert layout == "unknown"

    async def test_layout_cached_after_first_call(self, lhd_connector, mocker):
        mock_fs = mocker.MagicMock(
            ls=mocker.MagicMock(return_value=["nifs-lhd/1998"])
        )
        mocker.patch.object(lhd_connector, "_get_fs", return_value=mock_fs)

        await lhd_connector._discover_layout()
        await lhd_connector._discover_layout()

        # ls should only be called once despite two invocations
        assert mock_fs.ls.call_count == 1


class TestResolveSignal:
    def test_canonical_name(self, lhd_connector):
        group, var = lhd_connector._resolve_signal("plasma_current")
        assert group == "magnetics"
        assert var == "ip"

    def test_native_slash_format(self, lhd_connector):
        group, var = lhd_connector._resolve_signal("thomson/ne")
        assert group == "thomson"
        assert var == "ne"

    def test_unknown_raises_key_error(self, lhd_connector):
        with pytest.raises(KeyError):
            lhd_connector._resolve_signal("nonsense")


class TestClose:
    async def test_close_releases_fs(self, lhd_connector, mocker):
        lhd_connector._fs = mocker.MagicMock()  # simulate an open fs
        await lhd_connector.close()
        assert lhd_connector._fs is None
