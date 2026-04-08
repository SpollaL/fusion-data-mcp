"""Structured error responses for MCP tool handlers."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    UNKNOWN_MACHINE = "UNKNOWN_MACHINE"
    SHOT_NOT_FOUND = "SHOT_NOT_FOUND"
    SIGNAL_NOT_FOUND = "SIGNAL_NOT_FOUND"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    INVALID_PARAMS = "INVALID_PARAMS"
    TIMEOUT = "TIMEOUT"
    INTERNAL = "INTERNAL"


def error_response(code: ErrorCode, message: str, **extra) -> dict:
    return {"error": True, "code": str(code), "message": message, **extra}
