"""Bounded single-request stdio worker used by the SR native transport adapter."""
from __future__ import annotations

import argparse
import json
import sys

from pydantic import ValidationError

from alp_schema_mcp.runtime.catalog import ServerConfig
from .cloud import CloudAdapter, strict_loads
from alp_schema_mcp.runtime.errors import ALPError

MAX_IPC_BYTES = 16 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--projection", choices=["api_json", "typed"], default="api_json")
    parser.add_argument("--request-strict", action="store_true",
                        help="Request provider strict mode for typed schemas; actual enforcement is not assumed")
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(MAX_IPC_BYTES + 1)
        if len(raw) > MAX_IPC_BYTES:
            raise ALPError("IPC_TOO_LARGE", "The ALP worker request exceeds its bound.", 413)
        message = strict_loads(raw.decode())
        adapter = CloudAdapter(ServerConfig.from_file(args.config), projection=args.projection,
                               request_strict=args.request_strict)
        if message["action"] == "prepare":
            result = adapter.prepare(message["body"], message.get("headers"), message.get("history"))
        elif message["action"] == "complete":
            result = adapter.complete(message["context"], message["body"],
                                      allow_stop_tool_call=message.get("allow_stop_tool_call", False))
        else:
            raise ALPError("INVALID_WORKER_ACTION", "Unknown worker action.")
        output = {"ok": True, **result}
    except ALPError as exc:
        output = {"ok": False, "status": exc.status, **exc.envelope()}
    except (ValueError, ValidationError, KeyError, TypeError, RecursionError, UnicodeError):
        output = {"ok": False, "status": 400, "error": {"code": "INVALID_REQUEST", "message": "Invalid ALP request."}}
    encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_IPC_BYTES:
        encoded = b'{"ok":false,"status":502,"error":{"code":"IPC_TOO_LARGE","message":"Worker output exceeds its bound."}}'
    sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
