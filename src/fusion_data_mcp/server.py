"""
fusion-data-mcp — MCP server for nuclear fusion experimental databases.

Supported machines:
  - lhd   : Large Helical Device (NIFS, Japan) — AWS S3, no auth
  - mast  : Mega Amp Spherical Tokamak (UKAEA, UK) — FAIR-MAST REST API
  - cmod  : Alcator C-Mod (MIT, USA) — MDSplus public archive

Run locally:
  fusion-data-mcp

Add to Claude Desktop ~/.claude/claude_desktop_config.json:
  {
    "mcpServers": {
      "fusion-data": {
        "command": "fusion-data-mcp"
      }
    }
  }
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions
import mcp.server.stdio
import mcp.types as types

from .config import config
from .connectors.cmod import CModConnector
from .connectors.lhd import LHDConnector
from .connectors.mast import MASTConnector
from .errors import ErrorCode, error_response
from .models import ShotSearchParams
from .registry import ConnectorRegistry, UnknownMachineError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fusion-data-mcp")

# ---------------------------------------------------------------------------
# Registry — built once at module level, populated in lifespan
# ---------------------------------------------------------------------------
registry = ConnectorRegistry()
server = Server("fusion-data-mcp")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_devices",
            description=(
                "List all supported fusion devices with their capabilities, "
                "data sources, and access information."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="search_shots",
            description=(
                "Search for plasma shots (discharges) on a given device. "
                "Filter by date range, plasma current, density, or duration. "
                "Returns a list of matching shots with basic metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "Machine identifier: 'lhd', 'mast', or 'cmod'",
                    },
                    "date_from": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Start of date range (ISO 8601)",
                    },
                    "date_to": {
                        "type": "string",
                        "format": "date-time",
                        "description": "End of date range (ISO 8601)",
                    },
                    "min_plasma_current_MA": {
                        "type": "number",
                        "description": "Minimum plasma current in megaamperes",
                    },
                    "max_plasma_current_MA": {
                        "type": "number",
                        "description": "Maximum plasma current in megaamperes",
                    },
                    "min_duration_s": {
                        "type": "number",
                        "description": "Minimum shot duration in seconds",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["good", "disrupted", "incomplete", "unknown"],
                        "description": "Filter by shot status",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of results",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Pagination offset",
                    },
                },
                "required": ["device_id"],
            },
        ),
        types.Tool(
            name="get_shot_metadata",
            description=(
                "Get full metadata for a specific plasma shot. "
                "shot_id format: '{device_id}:{native_shot_id}', e.g. 'cmod:1120815012', "
                "'mast:30420', 'lhd:184255'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "shot_id": {
                        "type": "string",
                        "description": "Canonical shot ID: '{device}:{native_id}'",
                    },
                },
                "required": ["shot_id"],
            },
        ),
        types.Tool(
            name="list_diagnostics",
            description=(
                "List available diagnostics (signals) for a device or a specific shot. "
                "Use this to discover what data is available before calling get_signal."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "Machine identifier: 'lhd', 'mast', or 'cmod'",
                    },
                    "shot_id": {
                        "type": "string",
                        "description": "Optional: filter to diagnostics available for this shot",
                    },
                },
                "required": ["device_id"],
            },
        ),
        types.Tool(
            name="describe_signal",
            description=(
                "Get a statistical summary (min, max, mean, std, duration, sample count) "
                "of a signal without retrieving the full time series. "
                "Use this before get_signal to understand data quality and size."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "shot_id": {
                        "type": "string",
                        "description": "Canonical shot ID: '{device}:{native_id}'",
                    },
                    "diagnostic": {
                        "type": "string",
                        "description": (
                            "Canonical diagnostic name (e.g. 'plasma_current', "
                            "'electron_temperature') or native name"
                        ),
                    },
                },
                "required": ["shot_id", "diagnostic"],
            },
        ),
        types.Tool(
            name="get_signal",
            description=(
                "Retrieve a time-series signal from a plasma shot. "
                "Returns time axis (seconds) and data values. "
                "Large signals are automatically downsampled to max_samples points. "
                "Use describe_signal first to check signal quality."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "shot_id": {
                        "type": "string",
                        "description": "Canonical shot ID: '{device}:{native_id}'",
                    },
                    "diagnostic": {
                        "type": "string",
                        "description": "Canonical or native diagnostic name",
                    },
                    "t_start": {
                        "type": "number",
                        "description": "Start time in seconds (optional)",
                    },
                    "t_end": {
                        "type": "number",
                        "description": "End time in seconds (optional)",
                    },
                    "max_samples": {
                        "type": "integer",
                        "default": 10000,
                        "description": "Maximum number of time points to return",
                    },
                },
                "required": ["shot_id", "diagnostic"],
            },
        ),
        types.Tool(
            name="get_equilibrium",
            description=(
                "Retrieve MHD equilibrium reconstruction data for a plasma shot. "
                "Returns flux surface geometry, safety factor profile, and global parameters "
                "(plasma current, beta, internal inductance). "
                "Uses EFIT for tokamaks, VMEC for stellarators."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "shot_id": {
                        "type": "string",
                        "description": "Canonical shot ID: '{device}:{native_id}'",
                    },
                },
                "required": ["shot_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    try:
        result = await _dispatch(name, arguments)
    except UnknownMachineError as e:
        result = error_response(
            ErrorCode.UNKNOWN_MACHINE,
            str(e),
            available_devices=e.available,
        )
    except KeyError as e:
        result = error_response(ErrorCode.SIGNAL_NOT_FOUND, str(e))
    except Exception as e:
        logger.exception("Unhandled error in tool '%s'", name)
        result = error_response(ErrorCode.INTERNAL, str(e))

    import json
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _dispatch(name: str, args: dict[str, Any]) -> dict:
    if name == "list_devices":
        return await _list_devices()
    if name == "search_shots":
        return await _search_shots(args)
    if name == "get_shot_metadata":
        return await _get_shot_metadata(args)
    if name == "list_diagnostics":
        return await _list_diagnostics(args)
    if name == "describe_signal":
        return await _describe_signal(args)
    if name == "get_signal":
        return await _get_signal(args)
    if name == "get_equilibrium":
        return await _get_equilibrium(args)
    return error_response(ErrorCode.INTERNAL, f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _list_devices() -> dict:
    devices = [c.device_info.model_dump(mode="json") for c in registry.list_all()]
    return {"devices": devices, "total": len(devices)}


async def _search_shots(args: dict) -> dict:
    device_id = args["device_id"]
    connector = registry.get(device_id)

    from datetime import datetime
    def _parse_dt(v):
        return datetime.fromisoformat(v) if v else None

    params = ShotSearchParams(
        device_id=device_id,
        date_from=_parse_dt(args.get("date_from")),
        date_to=_parse_dt(args.get("date_to")),
        min_plasma_current_MA=args.get("min_plasma_current_MA"),
        max_plasma_current_MA=args.get("max_plasma_current_MA"),
        min_duration_s=args.get("min_duration_s"),
        status=args.get("status"),
        limit=args.get("limit", 20),
        offset=args.get("offset", 0),
    )
    shots = await connector.search_shots(params)
    return {"shots": [s.model_dump(mode="json") for s in shots], "total": len(shots)}


async def _get_shot_metadata(args: dict) -> dict:
    shot_id = args["shot_id"]
    connector = registry.get_for_shot(shot_id)
    meta = await connector.get_shot_metadata(shot_id)
    return meta.model_dump(mode="json")


async def _list_diagnostics(args: dict) -> dict:
    device_id = args["device_id"]
    shot_id = args.get("shot_id")
    connector = registry.get(device_id)
    result = await connector.list_diagnostics(shot_id)
    return result.model_dump(mode="json")


async def _describe_signal(args: dict) -> dict:
    shot_id = args["shot_id"]
    diagnostic = args["diagnostic"]
    connector = registry.get_for_shot(shot_id)
    summary = await connector.describe_signal(shot_id, diagnostic)
    return summary.model_dump(mode="json")


async def _get_signal(args: dict) -> dict:
    shot_id = args["shot_id"]
    diagnostic = args["diagnostic"]
    connector = registry.get_for_shot(shot_id)
    signal = await connector.get_signal(
        shot_id,
        diagnostic,
        t_start=args.get("t_start"),
        t_end=args.get("t_end"),
        max_samples=args.get("max_samples", config.default_max_samples),
    )
    return signal.model_dump(mode="json")


async def _get_equilibrium(args: dict) -> dict:
    shot_id = args["shot_id"]
    connector = registry.get_for_shot(shot_id)
    eq = await connector.get_equilibrium(shot_id)
    if eq is None:
        return error_response(
            ErrorCode.SIGNAL_NOT_FOUND,
            f"Equilibrium data not available for shot {shot_id}",
        )
    return eq.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def _run() -> None:
    # Register connectors based on config
    if config.enable_cmod:
        registry.register(CModConnector(pool_size=config.cmod_pool_size))
        logger.info("Registered connector: cmod")
    if config.enable_mast:
        registry.register(MASTConnector())
        logger.info("Registered connector: mast")
    if config.enable_lhd:
        registry.register(LHDConnector())
        logger.info("Registered connector: lhd")

    logger.info("fusion-data-mcp starting (transport=%s)", config.transport)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="fusion-data-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    import asyncio
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        loop.run_until_complete(registry.close_all())
        loop.close()


if __name__ == "__main__":
    main()
