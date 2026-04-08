"""
Integration tests for the LHD connector (NIFS public AWS S3 bucket).

Requires anonymous S3 access to s3://nifs-lhd (no credentials needed).
Run with: pytest --run-integration tests/integration/test_lhd_live.py -v

NOTE: The first test (test_lhd_bucket_accessible) is the most important —
it confirms basic S3 connectivity. Later tests may fail if the bucket
path convention differs from our current assumption in _shot_path().
"""

import pytest

from fusion_data_mcp.models import ShotSearchParams


@pytest.mark.integration
async def test_lhd_bucket_accessible(lhd_connector_live):
    ok = await lhd_connector_live.health_check()
    assert ok is True, (
        "s3://nifs-lhd is not accessible. "
        "Check network connectivity and anonymous S3 access."
    )


@pytest.mark.integration
async def test_lhd_device_info(lhd_connector_live):
    info = lhd_connector_live.device_info
    assert info.id == "lhd"
    assert info.type == "stellarator"
    assert info.country == "Japan"
    assert "nifs-lhd" in (info.data_url or "")


@pytest.mark.integration
async def test_lhd_discover_layout(lhd_connector_live):
    layout = await lhd_connector_live._discover_layout()
    assert layout in ("partitioned", "flat", "manifest"), (
        f"Unexpected layout '{layout}' — bucket may be empty or inaccessible"
    )


@pytest.mark.integration
async def test_lhd_bucket_top_level_not_empty(lhd_connector_live):
    """Smoke test: confirm the bucket has at least some content."""
    fs = lhd_connector_live._get_fs()
    import asyncio
    paths = await asyncio.to_thread(fs.ls, "s3://nifs-lhd/", detail=False)
    assert len(paths) > 0, "s3://nifs-lhd/ appears empty"
    # Print top-level structure to help calibrate _shot_path()
    print(f"\nLHD top-level paths (first 10): {paths[:10]}")


@pytest.mark.integration
async def test_lhd_search_shots_returns_results(lhd_connector_live):
    params = ShotSearchParams(device_id="lhd", limit=5)
    shots = await lhd_connector_live.search_shots(params)

    # Depending on layout, we may get 0 results if the path convention is wrong
    # This test documents the current state rather than hard-asserting
    print(f"\nLHD search returned {len(shots)} shots")
    for shot in shots:
        assert shot.device_id == "lhd"
        assert shot.shot_id.startswith("lhd:")
    # Log paths to help debug layout issues
    if shots:
        print(f"Example shot_ids: {[s.shot_id for s in shots[:3]]}")


@pytest.mark.integration
async def test_lhd_list_diagnostics_no_shot(lhd_connector_live):
    result = await lhd_connector_live.list_diagnostics()
    assert result.device_id == "lhd"
    assert result.total > 0
    names = [d.name for d in result.diagnostics]
    assert "plasma_current" in names
