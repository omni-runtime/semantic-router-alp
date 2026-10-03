from __future__ import annotations

import json

from alp_schema_mcp.catalog import ContractCatalog
from alp_schema_mcp.validation import (
    validate_action_exchange,
    validate_action_response,
    validate_canonical,
    validate_document,
)

from .catalog import stable_json
from .constraints import CompiledProfile
from .errors import ALPError
from .protocol import ALPChatRequest
from .schema_tools import compact_schema, generation_views
from .semantics import generation_guidance


def catalog_context(profile: CompiledProfile, *, include_effects=True) -> dict:
    """Describe visible targets and host-owned task constraints."""
    context = {}
    operations = set(profile.operations)
    if operations & {"agent_call", "agent_capability_call", "list_agent_capabilities"}:
        context["agents"] = {
            name: {
                "description": agent.description,
                "capabilities": {n: c.description for n, c in agent.capabilities.items()},
                **({"state_version": agent.state_version} if include_effects and agent.state_version is not None else {}),
                **({"capability_effects": {n: c.model_dump(include={"state_effect", "external_effect"}, exclude_none=True)
                                          for n, c in agent.capabilities.items() if c.state_effect is not None or c.external_effect is not None}}
                   if include_effects else {}),
            }
            for name, agent in profile.catalog.agents.items()
        }
    if operations & {"tool_call", "agent_definition_generate"}:
        context["tools"] = {name: tool.description for name, tool in profile.catalog.tools.items()}
        if include_effects:
            context["tool_effects"] = {name: tool.model_dump(include={"state_effect", "external_effect"}, exclude_none=True)
                                      for name, tool in profile.catalog.tools.items()}
        if include_effects and profile.catalog.current_state_version is not None:
            context["current_state_version"] = profile.catalog.current_state_version
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
    if profile.catalog.response_constraints:
        context["response_task_constraints"] = profile.catalog.response_constraints.model_dump(exclude_none=True)
    return context


def render_messages(
    request: ALPChatRequest, profile: CompiledProfile, contracts: ContractCatalog,
    *, codec: str = "tagged",
) -> list[dict]:
    if codec not in {"tagged", "canonical"}:
        raise ValueError("Unsupported ALP prompt codec")
    generation = profile.catalog.local_render_profile == "generation"
    definitions = []
    for operation in profile.operations:
        compact = profile.catalog.compact_prompt or (
            profile.catalog.compact_task_context and operation in profile.catalog.payload_constraints
        )
        views = generation_views(profile.body_schemas[operation], operation, codec=codec,
                                 flexible=profile.catalog.definition_field_order == "flexible",
                                 explicit_output=profile.catalog.explicit_definition_output,
                                 text_limit=profile.catalog.generation_text_limit,
                                 instruction_limit=profile.catalog.generation_instruction_limit)
        # JSON Schema ignores key order; alternatives differ only in that order.
        # Keep references rooted in their own document, not a new anyOf wrapper.
        view = views[0]
        if not generation:
            # The stable presentation labels the schema as a tag BODY and lists
            # operation separately. Keep its tested tokenization for small engines.
            view = profile.body_schemas[operation]
        definitions.append(
            {
                "operation": operation,
                **({"tag": contracts.binding(operation)["tag"]} if codec == "tagged" else {}),
                "body_schema": compact_schema(view, annotations=False) if compact else view,
                **({"generation_field_orders": [list(v["properties"]["payload"]["properties"]) for v in views]}
                   if generation and len(views) > 1 else {}),
            }
        )
    # Semantic presentation complements, but never replaces, the same generation
    # contract used by the decoder. No case-specific answers enter this view.
    context = {
        "protocol_version": request.alp.protocol_version,
        "model_output_codec": codec,
        "allowed_action_contracts": definitions,
        "visible_catalog": catalog_context(profile, include_effects=generation or
            profile.catalog.current_state_version is not None or
            any(agent.state_version is not None for agent in profile.catalog.agents.values())),
        **({"generation_guidance": generation_guidance(profile)} if generation else {}),
    }
    if contracts.version == "0.4.0":
        plan = profile.catalog.response_constraints
        context["response_collection"] = {"minItems": plan.min_calls if plan else 1, "maxItems": plan.max_calls if plan else 16,
            "unique_request_ids": True, "exclusive": ["agent_final", "resource.bindings.update"]}
    messages = [{
        "role": "system",
        "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
    }]
    pending, results = [], []
    terminal = False
    for message in request.messages:
        if terminal:
            raise ALPError("INVALID_HISTORY", "An ALP final terminates the session.")
        if message.agent_calls:
            if pending:
                raise ALPError("INVALID_HISTORY", "Every prior call requires its terminal result.")
            actions = [call.request for call in message.agent_calls]
            if contracts.version == "0.4.0":
                report = validate_action_response(stable_json(actions), "canonical", contracts)
            else:
                report = validate_canonical(actions[0], contracts)
            if not report["valid"] or any("payload" not in action or "status" in action for action in actions):
                raise ALPError("INVALID_HISTORY", "Assistant history must contain a valid ALP response.")
            if contracts.version == "0.4.0":
                pending = message.agent_calls
                results = []
                terminal = actions[0]["operation"] == "agent_final"
            messages.append({"role": "assistant", "content":
                "\n".join(contracts.representations(action)["tagged"] for action in actions)
                if codec == "tagged" else stable_json(actions if contracts.version == "0.4.0" else actions[0])})
        else:
            if contracts.version == "0.4.0":
                if pending:
                    call = pending[len(results)]
                    if message.role != "tool" or message.tool_call_id != call.id or not isinstance(message.content, str):
                        raise ALPError("INVALID_HISTORY", "Results must match all calls in original request order.")
                    report = validate_document(message.content, "canonical", contracts)
                    if not report["valid"]:
                        raise ALPError("INVALID_HISTORY", "Invalid terminal result.")
                    results.append(report["canonical"])
                    if len(results) == len(pending):
                        paired = validate_action_exchange([call.request for call in pending], results, contracts)
                        if not paired["valid"]:
                            raise ALPError("INVALID_HISTORY", "Invalid terminal result pairing.", 400, paired["errors"])
                        pending = []
                elif message.role == "tool":
                    raise ALPError("INVALID_HISTORY", "Result has no pending call.")
            messages.append(message.model_dump(exclude_none=True, exclude={"agent_calls"}))
    if contracts.version == "0.4.0" and (pending or terminal):
        raise ALPError("INVALID_HISTORY", "Next generation requires all terminal results and a non-final session.")
    return messages
