from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from alp_schema_mcp.catalog import ContractCatalog
from jsonschema import Draft202012Validator, SchemaError
from pydantic import Field, model_validator

from .errors import ALPError
from .protocol import Operation, StrictModel


def schema_children(node: dict):
    for key in ("properties", "patternProperties", "$defs", "dependentSchemas"):
        yield from node.get(key, {}).values()
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        yield from node.get(key, [])
    for key in (
        "items",
        "additionalProperties",
        "if",
        "then",
        "else",
        "not",
        "contains",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
    ):
        if key in node:
            yield node[key]


def check_schema(schema: dict | bool) -> None:
    """Check schemas and resolve all references locally, without network access."""
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError("Invalid JSON Schema") from exc
    pending = [schema]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {"$ref", "$dynamicRef"}:
                    if (
                        key == "$dynamicRef"
                        or not isinstance(value, str)
                        or not value.startswith("#/")
                    ):
                        raise ValueError("Only local JSON Pointer $ref is supported")
                    target: Any = schema
                    try:
                        for part in value[2:].split("/"):
                            part = part.replace("~1", "/").replace("~0", "~")
                            target = target[int(part)] if isinstance(target, list) else target[part]
                    except (KeyError, ValueError, TypeError, IndexError) as exc:
                        raise ValueError("Unresolved local schema reference") from exc
                    if not isinstance(target, (dict, bool)):
                        raise ValueError("A reference must resolve to a schema")
            pending.extend(schema_children(item))


@lru_cache(maxsize=1)
def _identifier_validators():
    contracts = ContractCatalog()
    return {
        name: Draft202012Validator(contracts.schema("protocol", name))
        for name in ("Id", "Name", "CatalogRef", "ReleaseVersion")
    }


class Capability(StrictModel):
    description: str = ""
    input_schema: dict[str, Any]


class Agent(StrictModel):
    description: str = ""
    input_schema: dict[str, Any]
    capabilities: dict[str, Capability] = Field(default_factory=dict)


class Tool(StrictModel):
    description: str = ""
    input_schema: dict[str, Any]
    external_effect: Literal["none", "read", "write"] | None = None
    state_effect: Literal["read", "write"] = "read"


class CapabilityContract(StrictModel):
    """Exact interface requirements for one named generated capability."""

    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    state_effect: Literal["read", "write"] | None = None
    external_effect: Literal["none", "read", "write"] | None = None


class ResourceBinding(StrictModel):
    kind: Literal["knowledge", "skill", "mcp", "asset"]
    slot: str
    profile_ref: str
    version: str | None = None
    tool_allowlist: list[str] | None = None


class HandlerContract(StrictModel):
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    state_schema: dict[str, Any]
    state_effect: Literal["read", "write"]
    external_effect: Literal["none", "read", "write"]
    required_tools: list[str] = Field(default_factory=list)


class PayloadConstraints(StrictModel):
    """Operator-owned constraints, addressed by object pointers under payload."""

    required_fields: list[str] = Field(default_factory=list, max_length=64)
    fixed_values: dict[str, Any] = Field(default_factory=dict, max_length=64)
    capabilities: dict[str, CapabilityContract] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def check_paths(self):
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("Constraint paths must be unique")
        for path in [*self.required_fields, *self.fixed_values]:
            if not path.startswith("/") or len(path) > 512 or len(path.split("/")) > 9:
                raise ValueError("Use object JSON Pointers relative to payload, depth at most 8")
            if any(not part or "~" in part.replace("~0", "").replace("~1", "")
                   for part in path[1:].split("/")):
                raise ValueError("Invalid object JSON Pointer")
        json.dumps(self.fixed_values, allow_nan=False)
        contracts = ContractCatalog()
        for name, capability in self.capabilities.items():
            if not _identifier_validators()["Name"].is_valid(name):
                raise ValueError("Invalid capability name")
            for field, schema in (("input_schema", capability.input_schema), ("output_schema", capability.output_schema)):
                if schema is not None and not Draft202012Validator(contracts.schema(
                    "protocol", "ObjectSchema" if field == "input_schema" else "ValueSchema"
                )).is_valid(schema):
                    raise ValueError("Capability contract must use an ALP value schema")
        return self


class Catalog(StrictModel):
    allowed_operations: list[Operation]
    agents: dict[str, Agent] = Field(default_factory=dict)
    tools: dict[str, Tool] = Field(default_factory=dict)
    final_format: Literal["text", "json"] = "text"
    final_schema: dict[str, Any] | None = None
    payload_constraints: dict[Operation, PayloadConstraints] = Field(default_factory=dict)
    definition_field_order: Literal["contract", "flexible"] = "contract"
    explicit_definition_output: bool = False
    explicit_session_mode: bool = False
    compact_prompt: bool = False
    compact_task_context: bool = True
    generation_text_limit: int = Field(default=0, ge=0, le=16000)
    generation_instruction_limit: int = Field(default=0, ge=0, le=16000)
    # None means the host did not provide this domain; [] means no choices.
    environment_profiles: list[str] | None = None
    resource_bindings: list[ResourceBinding] | None = None
    evidence_bindings: dict[str, str] | None = None
    handlers: dict[str, HandlerContract] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_contracts(self):
        if not self.allowed_operations or len(set(self.allowed_operations)) != len(
            self.allowed_operations
        ):
            raise ValueError("Catalog operations must be nonempty and unique")
        if not set(self.payload_constraints).issubset(self.allowed_operations):
            raise ValueError("Payload constraints must target catalog operations")
        if any(c.capabilities and op != "agent_definition_generate" for op, c in self.payload_constraints.items()):
            raise ValueError("Named capability contracts apply only to definition generation")
        identifiers = _identifier_validators()
        contracts = ContractCatalog()
        for name, tool in self.tools.items():
            reserved = contracts.resource_tools.get(name)
            if reserved and (tool.external_effect not in (None, reserved["external_effect"]) or tool.state_effect != reserved["state_effect"]):
                raise ValueError("Reserved tool effects cannot be overridden")
        refs = [*(self.environment_profiles or []), *self.handlers]
        for binding in self.resource_bindings or []:
            refs.extend((binding.slot, binding.profile_ref))
            if binding.kind in {"skill", "asset"} and binding.version is None:
                raise ValueError("Versioned resources require a version")
            if binding.version is not None and not identifiers["ReleaseVersion"].is_valid(binding.version):
                raise ValueError("Invalid resource version")
            if binding.kind == "mcp" and not binding.tool_allowlist:
                raise ValueError("MCP resources require tools")
            if binding.kind != "mcp" and binding.tool_allowlist is not None:
                raise ValueError("Only MCP resources have a tool allowlist")
            if binding.kind not in {"skill", "asset"} and binding.version is not None:
                raise ValueError("Only skills and assets have a version")
            if not set(binding.tool_allowlist or []).issubset(self.tools):
                raise ValueError("Resource tools must exist in the catalog")
        slots = [r.slot for r in self.resource_bindings or []]
        if len(slots) != len(set(slots)):
            raise ValueError("Resource binding slots must be unique")
        if any(not identifiers["CatalogRef"].is_valid(ref) for ref in refs):
            raise ValueError("Invalid host catalog reference")
        if self.evidence_bindings is not None:
            knowledge_slots = {r.slot for r in self.resource_bindings or [] if r.kind == "knowledge"}
            if any(not identifiers["Id"].is_valid(e) or slot not in knowledge_slots for e, slot in self.evidence_bindings.items()):
                raise ValueError("Evidence must reference a known knowledge binding")
        for handler in self.handlers.values():
            for field in ("input_schema", "output_schema", "state_schema"):
                schema = getattr(handler, field)
                kind = "ValueSchema" if field == "output_schema" else "ObjectSchema"
                if not Draft202012Validator(contracts.schema("protocol", kind)).is_valid(schema):
                    raise ValueError("Handler contracts must use ALP value schemas")
            if not set(handler.required_tools).issubset(self.tools):
                raise ValueError("Handler tools must exist in the catalog")
        if any(not identifiers["CatalogRef"].is_valid(name) for name in self.tools):
            raise ValueError("Tool names must satisfy ALP CatalogRef")
        schemas = [tool.input_schema for tool in self.tools.values()]
        for name, agent in self.agents.items():
            if not identifiers["Id"].is_valid(name):
                raise ValueError("Agent IDs must satisfy ALP Id")
            if any(not identifiers["Name"].is_valid(cap_name) for cap_name in agent.capabilities):
                raise ValueError("Capability names must satisfy ALP Name")
            schemas.append(agent.input_schema)
            schemas.extend(cap.input_schema for cap in agent.capabilities.values())
        for schema in schemas:
            if schema.get("type") != "object":
                raise ValueError("Input schemas must describe objects")
            check_schema(schema)
        if self.final_format == "json":
            if self.final_schema is None:
                raise ValueError("JSON final output requires final_schema")
            check_schema(self.final_schema)
        elif self.final_schema is not None:
            raise ValueError("Text output does not use final_schema")
        return self


class ServerConfig(StrictModel):
    catalogs: dict[str, Catalog] = Field(min_length=1)
    backend: Literal["auto", "vllm", "omni"] = "auto"
    compiler_cache_size: int = Field(default=64, ge=1, le=1024)
    max_request_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    # Only server operators configure model-specific reasoning/template controls.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=lambda: {"enable_thinking": False})
    allow_reasoning: bool = False
    task_signing_key_env: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> ServerConfig:
        try:
            raw = Path(path).read_bytes()
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Configuration exceeds 4 MiB")
            return cls.model_validate_json(raw)
        except (OSError, ValueError) as exc:
            raise ALPError(
                "INVALID_CONFIG", "Cannot load a valid ALP server configuration.", 503
            ) from exc


class CatalogRegistry:
    """Deployment-scoped catalogs. Override resolve for tenant-aware policy."""

    def __init__(self, config: ServerConfig):
        self.config = config.model_copy(deep=True)

    async def resolve(self, ref: str, raw_request: Any = None) -> Catalog:
        catalog = self.config.catalogs.get(ref)
        if catalog is None:
            raise ALPError("UNKNOWN_CATALOG", "The requested ALP catalog is unavailable.", 404)
        return catalog.model_copy(deep=True)


def stable_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
