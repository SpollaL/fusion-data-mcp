# fusion-data-mcp

An MCP server that gives AI assistants direct access to nuclear fusion experimental databases.

Ask Claude things like:
> *"Find MAST shots from 2010 with plasma current above 0.5 MA and show the stored energy evolution"*
> *"Compare the electron temperature profiles between these two LHD shots"*
> *"Summarise the disruption signatures in C-Mod shot 1120815012"*

## Supported databases

| Device | Type | Access | Coverage |
|---|---|---|---|
| **LHD** (NIFS, Japan) | Stellarator | AWS S3 — anonymous | 1998–present, ~2 PB |
| **MAST** (UKAEA, UK) | Tokamak | FAIR-MAST REST API + Zarr | Campaigns M05–M09 |
| **Alcator C-Mod** (MIT, USA) | Tokamak | MDSplus public archive | 1991–2016 |

All three are publicly accessible with no credentials required (C-Mod requires port 8000 to be reachable — see [notes](#alcator-c-mod)).

## MCP tools

| Tool | Description |
|---|---|
| `list_devices` | List supported machines with capabilities and data sources |
| `search_shots` | Find shots by date range, plasma current, density, status |
| `get_shot_metadata` | Full metadata for a specific shot |
| `list_diagnostics` | Available diagnostics for a device or shot |
| `describe_signal` | Statistical summary (min/max/mean/std) without loading full data |
| `get_signal` | Retrieve a time-series signal (auto-downsampled to `max_samples`) |
| `get_equilibrium` | MHD equilibrium reconstruction (EFIT for tokamaks, VMEC for LHD) |

Shot IDs use the format `{device}:{native_id}` — e.g. `mast:30420`, `lhd:184255`, `cmod:1120815012`.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SpollaL/fusion-data-mcp
cd fusion-data-mcp
uv sync
```

## Usage with Claude Desktop

Add to your `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fusion-data": {
      "command": "uv",
      "args": ["--directory", "/path/to/fusion-data-mcp", "run", "fusion-data-mcp"]
    }
  }
}
```

Restart Claude Desktop. The `list_devices` tool will confirm which backends are reachable.

## Usage with Claude Code

```bash
claude mcp add fusion-data -- uv --directory /path/to/fusion-data-mcp run fusion-data-mcp
```

## Configuration

All settings are via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | `stdio` for local use, `sse` for remote |
| `ENABLE_CMOD` | `true` | Enable/disable C-Mod connector |
| `ENABLE_MAST` | `true` | Enable/disable MAST connector |
| `ENABLE_LHD` | `true` | Enable/disable LHD connector |
| `CMOD_POOL_SIZE` | `3` | MDSplus connection pool size |
| `DEFAULT_MAX_SAMPLES` | `10000` | Default signal downsampling limit |

## Alcator C-Mod

C-Mod data is publicly accessible via MDSplus (`cmodpub`/`cmodpub`) but requires **port 8000** to be open between your machine and `alcdata.psfc.mit.edu`. This port is blocked on most home and cloud networks.

It works from MIT campus/VPN, university networks with academic peering, or any machine that can reach `alcdata.psfc.mit.edu:8000`. C-Mod tests are automatically skipped if the port is unreachable — the other two backends are unaffected.

## Development

```bash
uv sync --extra dev

# Unit tests (no network required)
uv run pytest tests/unit/ -v

# Integration tests (live network)
uv run pytest tests/integration/ --run-integration -v

# All tests
RUN_INTEGRATION_TESTS=1 uv run pytest -v
```

## Project structure

```
src/fusion_data_mcp/
├── server.py          # MCP server entrypoint
├── registry.py        # Routes requests to the right connector
├── serialization.py   # numpy/xarray → JSON-safe conversion
├── config.py          # Environment variable config
├── models/            # Pydantic schemas (Shot, Signal, Diagnostic, Equilibrium)
└── connectors/
    ├── base.py        # AbstractConnector interface
    ├── cmod.py        # MDSplus + async connection pool
    ├── mast.py        # FAIR-MAST REST API + Zarr
    └── lhd.py         # AWS S3 + HDF5
```

## Adding a new data source

Implement `AbstractConnector` in a new file under `connectors/`, then register it in `server.py`:

```python
# connectors/d3d.py
class D3DConnector(AbstractConnector):
    device_id = "d3d"
    ...

# server.py
registry.register(D3DConnector())
```

The connector must implement: `search_shots`, `get_shot_metadata`, `list_diagnostics`, `get_signal`, `describe_signal`, `get_equilibrium`, `health_check`.

## Data licenses

- **LHD**: [NIFS Rights and Terms](https://www-lhd.nifs.ac.jp/pub/RightsTerms.html)
- **MAST**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **C-Mod**: Public archive, MIT PSFC

## License

MIT — see [LICENSE](LICENSE).
