"""Request-bound constraints supplied by a trusted host, never by model output."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

from pydantic import Field, model_validator

from .catalog import (
    Catalog,
    PayloadConstraints,
    ResponseConstraints,
    ServerConfig,
    stable_json,
)
from .errors import ALPError
from .protocol import ALPChatRequest, Operation, StrictModel

TASK_HEADER = "x-alp-task-context"
MAX_HEADER_BYTES = 16384
MAX_TTL_SECONDS = 300
DOMAIN = b"alp-host-task-v1:"


class TaskContext(StrictModel):
    version: int = Field(default=1, ge=1, le=2)
    issued_at: int
    expires_at: int
    request_digest: str
    constraints: dict[Operation, PayloadConstraints] = Field(default_factory=dict, max_length=6)
    response: ResponseConstraints | None = None

    @model_validator(mode="after")
    def check_version(self):
        if not self.constraints and self.response is None:
            raise ValueError("Empty host constraints")
        if self.response is not None and self.version != 2:
            raise ValueError("Response constraints require host context v2")
        return self


def request_digest(request: ALPChatRequest) -> str:
    return hashlib.sha256(stable_json(request.model_dump(mode="json")).encode()).hexdigest()


def task_headers(
    request: ALPChatRequest,
    constraints: dict[Operation, PayloadConstraints],
    *,
    key: bytes,
    response: ResponseConstraints | None = None,
    ttl_seconds: int = 60,
    now: int | None = None,
) -> dict[str, str]:
    """Sign reviewed task values for this exact request; this grants no action permissions."""
    if len(key) < 32 or not 1 <= ttl_seconds <= MAX_TTL_SECONDS:
        raise ValueError("Use a key of at least 32 bytes and a TTL between 1 and 300 seconds")
    issued = int(time.time()) if now is None else now
    context = TaskContext(
        version=2 if response is not None else 1, response=response,
        issued_at=issued, expires_at=issued + ttl_seconds,
        request_digest=request_digest(request), constraints=constraints,
    )
    payload = base64.urlsafe_b64encode(stable_json(context.model_dump()).encode()).decode().rstrip("=")
    signature = hmac.new(key, DOMAIN + payload.encode(), hashlib.sha256).hexdigest()
    token = payload + "." + signature
    if len(token) > MAX_HEADER_BYTES:
        raise ValueError("Task context exceeds the header size limit")
    return {TASK_HEADER: token}


def bind_task_constraints(
    request: ALPChatRequest, catalog: Catalog, config: ServerConfig, raw_request=None,
) -> Catalog:
    if raw_request is None:
        return catalog
    values = raw_request.headers.getlist(TASK_HEADER)
    if not values:
        return catalog

    def reject():
        return ALPError("INVALID_TASK_CONTEXT", "The host task context is invalid or expired.", 403)

    if len(values) != 1 or len(values[0]) > MAX_HEADER_BYTES or not config.task_signing_key_env:
        raise reject()
    key = os.environ.get(config.task_signing_key_env, "").encode()
    if len(key) < 32:
        raise ALPError("TASK_CONTEXT_NOT_CONFIGURED", "The host task signing key is unavailable.", 503)
    try:
        payload, signature = values[0].split(".")
        expected = hmac.new(key, DOMAIN + payload.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid signature")
        encoded = payload + "=" * (-len(payload) % 4)
        context = TaskContext.model_validate_json(base64.b64decode(encoded, altchars=b"-_", validate=True))
        now = int(time.time())
        if not context.issued_at <= now < context.expires_at:
            raise ValueError("Expired context")
        if not 0 < context.expires_at - context.issued_at <= MAX_TTL_SECONDS:
            raise ValueError("Invalid lifetime")
        if not hmac.compare_digest(context.request_digest, request_digest(request)):
            raise ValueError("Different request")
        if not set(context.constraints).issubset(request.alp.allowed_operations):
            raise ValueError("Different operation")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise reject() from None

    bound = catalog.model_copy(deep=True)
    from .response_constraints import merge_payload, validate_plan
    try:
        for operation, extra in context.constraints.items():
            if operation not in catalog.allowed_operations:
                raise ValueError("Unavailable operation")
            bound.payload_constraints[operation] = merge_payload(
                bound.payload_constraints.get(operation, PayloadConstraints()), extra)
        if context.response is not None:
            base = bound.response_constraints
            extra = context.response
            if base is not None:
                if extra.min_calls < base.min_calls or extra.max_calls > base.max_calls:
                    raise ValueError("Host response bounds cannot be widened")
                if base.members:
                    if not extra.members or len(base.members) != len(extra.members):
                        raise ValueError("Host response members cannot be removed")
                    extra = extra.model_copy(deep=True)
                    for before, after in zip(base.members, extra.members, strict=True):
                        if before.operation != after.operation:
                            raise ValueError("Host response operation cannot change")
                        after.payload = merge_payload(before.payload, after.payload)
            bound.response_constraints = extra
        operations = request.alp.allowed_operations
        if request.alp.choice != "required":
            operations = [request.alp.choice.operation]
        validate_plan(request.alp.protocol_version, operations, bound)
    except (ValueError, ALPError):
        raise reject() from None
    return bound
