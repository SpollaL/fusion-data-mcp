# Contributing

Contributions are welcome — bug fixes, new connectors, better signal mappings, and documentation improvements.

## Getting started

```bash
git clone https://github.com/SpollaL/fusion-data-mcp
cd fusion-data-mcp
uv sync --extra dev
```

Run the test suite before making any changes to confirm everything is green:

```bash
uv run pytest tests/unit/ -v
```

## What to work on

Check the [issues](https://github.com/SpollaL/fusion-data-mcp/issues) for open tasks. Good first contributions:

- **Signal mappings** — the canonical diagnostic maps in each connector are incomplete. If you know the native name of a signal on LHD, MAST, or C-Mod, add it.
- **New connectors** — DIII-D, NSTX-U, W7-X, EAST are the obvious next targets. See below.
- **LHD path convention** — the bucket layout probe in `lhd.py` makes assumptions. Real LHD data access experience would help verify and improve it.

## Adding a connector

1. Create `src/fusion_data_mcp/connectors/{device_id}.py`
2. Implement `AbstractConnector` — all 7 abstract methods must be async
3. Register it in `server.py` under the appropriate `config.enable_*` flag
4. Add integration tests in `tests/integration/test_{device_id}_live.py`
5. Add unit tests for `_resolve_signal`, `_parse_shot`/equivalent, and any non-trivial internal logic

The key constraint: **all blocking I/O must be wrapped in `asyncio.to_thread()`**. This includes MDSplus, h5py, and any synchronous HTTP client. See `cmod.py` for the connection pool pattern and `mast.py` for the httpx async pattern.

## Signal naming convention

Canonical diagnostic names (defined in `models/diagnostic.py`) must be machine-agnostic:

- `plasma_current` not `ip` or `IP` or `Ip`
- `electron_temperature` not `Te` or `T_e`
- Units are in SI with appropriate scale: current in MA, temperature in keV, density in m⁻³

If a signal doesn't map cleanly to a canonical name, keep it native (`source/name` for MAST, `TREE::NODE` for MDSplus) and leave `canonical_name=None`.

## Tests

- **Unit tests** must run without network access. Mock all I/O.
- **Integration tests** require live data — mark with `@pytest.mark.integration` and use a fixture from `tests/integration/conftest.py`.
- Don't test obvious Python/Pydantic behaviour. Test the logic you actually wrote.

```bash
# Unit only (fast, no network)
uv run pytest tests/unit/ -v

# Integration (needs network)
uv run pytest tests/integration/ --run-integration -v
```

## Code style

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

No docstrings required on internal methods. Keep comments to where the logic isn't self-evident (MDSplus node paths, Zarr layout assumptions, async wrapping decisions).

## Pull requests

- One logical change per PR
- All unit tests must pass
- If you're adding a connector, at least the `health_check` integration test should pass
- Update `README.md` if you're adding a supported device

## Reporting issues

If you have access to a fusion data system and find that a signal path or API endpoint is wrong, open an issue with:
- The device
- The signal name (canonical and native)
- The correct path/endpoint
- A shot number where it can be verified

This is the most valuable contribution you can make — domain knowledge is the bottleneck.
