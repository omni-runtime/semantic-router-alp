from __future__ import annotations

from typing import Any


class ALPError(Exception):
    """A sanitized, public error. Never include model output or credentials."""

    def __init__(
        self, code: str, message: str, status: int = 400, details: list[dict] | None = None
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or []

    def envelope(self) -> dict[str, Any]:
        return {
            "error": {
                "type": "alp_error",
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }
