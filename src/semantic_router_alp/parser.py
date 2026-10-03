from __future__ import annotations

from typing import Any

from alp_schema_mcp.catalog import ContractCatalog
from alp_schema_mcp.validation import MAX_JSON_DEPTH, MAX_REQUEST_BYTES, validate_document
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry
from referencing.exceptions import NoSuchResource

from .constraints import CompiledProfile
from .errors import ALPError


def _deny_remote(uri):
    raise NoSuchResource(ref=uri)


class ALPParser:
    """Request-scoped incremental framing. Only finish() accepts an action."""

    def __init__(self, profile: CompiledProfile, contracts: ContractCatalog, codec="tagged"):
        if codec not in {"tagged", "canonical"}:
            raise ValueError("Unsupported model output codec")
        self.profile = profile
        self.contracts = contracts
        self.codec = codec
        self._parts: list[str] = []
        self._size = 0
        self._finished = False
        self._depth = 0
        self._in_string = False
        self._escaped = False

    @property
    def raw(self) -> str:
        return "".join(self._parts)

    def feed(self, text: str) -> None:
        if self._finished:
            raise ALPError(
                "OUTPUT_AFTER_FINISH", "Output arrived after the completion boundary.", 502
            )
        try:
            self._size += len(text.encode("utf-8"))
        except UnicodeError as exc:
            raise ALPError(
                "INVALID_UNICODE", "The model output is not valid Unicode.", 502
            ) from exc
        if self._size > MAX_REQUEST_BYTES:
            raise ALPError("OUTPUT_TOO_LARGE", "The ALP output exceeds 128 KiB.", 502)
        for char in text:
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char in "[{":
                self._depth += 1
                if self._depth > MAX_JSON_DEPTH:
                    raise ALPError(
                        "OUTPUT_TOO_DEEP", "The model output exceeds the ALP nesting limit.", 502
                    )
            elif char in "]}":
                self._depth -= 1
        self._parts.append(text)

    def finish(self, reason: str | None) -> dict[str, Any]:
        if self._finished:
            raise ALPError("DUPLICATE_FINISH", "A completion can be accepted only once.", 502)
        self._finished = True
        if reason != "stop":
            raise ALPError(
                "INCOMPLETE_GENERATION", "The model did not complete a normal ALP generation.", 502
            )
        report = validate_document(self.raw, self.codec, self.contracts)
        if not report["valid"]:
            raise ALPError(
                "INVALID_ALP_OUTPUT",
                "The model output failed ALP validation.",
                502,
                report["errors"],
            )
        canonical = report["canonical"]
        operation = canonical["operation"]
        if operation not in self.profile.operations:
            raise ALPError(
                "UNEXPECTED_OPERATION", "The model selected an unavailable operation.", 502
            )
        body = {key: canonical[key] for key in ("protocol_version", "request_id", "payload")}
        validator = Draft202012Validator(
            self.profile.body_schemas[operation],
            format_checker=FormatChecker(),
            registry=Registry(retrieve=_deny_remote),
        )
        errors = []
        for error in validator.iter_errors(body):
            # Do not return jsonschema's messages: they contain supplied values.
            errors.append(
                {
                    "code": "CATALOG_CONTRACT_MISMATCH",
                    "rule": str(error.validator),
                    "path": "/" + "/".join(str(x) for x in error.absolute_path),
                }
            )
            if len(errors) == 32:
                break
        if errors:
            raise ALPError(
                "INVALID_CALL_ARGUMENTS", "The call does not satisfy this catalog.", 502, errors
            )
        from .host_contracts import validate_host_contracts

        errors = validate_host_contracts(canonical, self.profile.catalog, self.contracts)
        if errors:
            raise ALPError("INVALID_HOST_CONTRACT", "The output violates a trusted host contract.", 502, errors[:32])
        return canonical
