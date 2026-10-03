from __future__ import annotations

import json

from alp_schema_mcp.catalog import ContractCatalog
from alp_schema_mcp.validation import validate_canonical

from .catalog import stable_json
from .constraints import CompiledProfile
from .errors import ALPError
from .protocol import ALPChatRequest
from .schema_tools import compact_schema


def catalog_context(profile: CompiledProfile) -> dict:
    """Describe visible targets and host-owned task constraints."""
    context = {}
    operations = set(profile.operations)
    if operations & {"agent_call", "agent_capability_call", "list_agent_capabilities"}:
        context["agents"] = {
            name: {
                "description": agent.description,
                "capabilities": {n: c.description for n, c in agent.capabilities.items()},
            }
            for name, agent in profile.catalog.agents.items()
        }
    if operations & {"tool_call", "agent_definition_generate"}:
        context["tools"] = {name: tool.description for name, tool in profile.catalog.tools.items()}
        for field in ("environment_profiles", "resource_bindings", "evidence_bindings", "handlers"):
            value = profile.catalog.model_dump(exclude_none=True).get(field)
            if value is not None:
                context[field] = value
    constraints = {
        op: value.model_dump() for op, value in profile.catalog.payload_constraints.items()
        if op in operations
    }
    if constraints:
        context["required_task_constraints"] = constraints
    return context


def render_messages(
    request: ALPChatRequest, profile: CompiledProfile, contracts: ContractCatalog,
    *, codec: str = "tagged",
) -> list[dict]:
    if codec not in {"tagged", "canonical"}:
        raise ValueError("Unsupported ALP prompt codec")
    definitions = []
    for operation in profile.operations:
        compact = profile.catalog.compact_prompt or (
            profile.catalog.compact_task_context and operation in profile.catalog.payload_constraints
        )
        definitions.append(
            {
                "operation": operation,
                **({"tag": contracts.binding(operation)["tag"]} if codec == "tagged" else {}),
                "body_schema": compact_schema(profile.body_schemas[operation], annotations=False)
                if compact else profile.body_schemas[operation],
            }
        )
    # Context describes the contract and visible targets. Generation constraints
    # enforce the wire format; do not add handwritten protocol rules or examples.
    context = {
        "protocol_version": request.alp.protocol_version,
        "model_output_codec": codec,
        "allowed_action_contracts": definitions,
        "visible_catalog": catalog_context(profile),
    }
    messages = [{
        "role": "system",
        "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
    }]
    for message in request.messages:
        if message.agent_calls:
            canonical = message.agent_calls[0].request
            report = validate_canonical(canonical, contracts)
            if (
                not report["valid"]
                or canonical.get("operation") not in {x["operation"] for x in contracts.bindings}
                or "status" in canonical
            ):
                raise ALPError(
                    "INVALID_HISTORY", "Assistant history must contain a valid ALP request."
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": contracts.representations(canonical)["tagged"]
                    if codec == "tagged" else stable_json(canonical),
                }
            )
        else:
            messages.append(message.model_dump(exclude_none=True, exclude={"agent_calls"}))
    return messages
