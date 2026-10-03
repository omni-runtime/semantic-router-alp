"""Protocol-derived presentation, shared by the bundled adapters.

Sources: frozen ALP README sections 5/7/9/10/18 and resource binding semantics.
These hints contain no task answers and never replace canonical validation.
"""
from __future__ import annotations

OPERATION_DESCRIPTIONS = {
    "agent_definition_generate": "Define an agent, its requested resources/tools and interfaces. Output schemas are interface contracts, not example answers.",
    "agent_call": "Send a task to an existing visible agent. Preserve explicitly verbatim input. session_mode defaults to continue; use isolated only when requested.",
    "agent_capability_call": "Invoke a visible named capability with its declared arguments. A state-writing capability requires the target's observed business-state version.",
    "list_agent_capabilities": "Inspect an existing visible agent's capabilities without inventing an instance ID.",
    "tool_call": "Call a visible tool with its declared arguments. Business-state writes require an observed expected_state_version; external writes alone do not.",
    "agent_final": "Finish with the required output format and value. This is the only action in its response.",
}


def generation_guidance(profile):
    if not profile.catalog.semantic_rendering:
        return {}
    rules = {
        "parameters": "Follow the current task. A schema lists allowed fields, not a request to populate all of them. For exact parameter encoding, preserve supplied values and omit unspecified substantive options; documented defaults may be omitted. Do not invent profiles, interfaces, state versions or resources.",
        "operations": {op: OPERATION_DESCRIPTIONS[op] for op in profile.operations},
    }
    if profile.protocol_version == "0.4.0":
        rules["collection"] = "Return all independent calls requested for this decision, including repeated uses of the same function. The maximum is a ceiling, not a target. Stop after completing the requested calls; preserve requested order and use distinct request IDs. Do not merge calls or wait for tool results between independent members. Never invent results or future instance IDs."
    if "tool_call" in profile.operations:
        rules["state_versions"] = "expected_state_version has no default. Omit it for reads unless a version guard is explicitly supplied. Never guess 1. Resource revision is separate from business-state version; preserve each supplied revision. resource.bindings.update must be the only action; preserve each requested add/replace/remove change and its resource kind."
    if "agent_definition_generate" in profile.operations:
        rules["definition"] = "Generate descriptions/instructions when the task leaves them free; preserve text explicitly requested verbatim. A requested JSON output contract must remain JSON with that schema. Tools and resource requirements are dependencies, not permission to add unrelated profiles, policies or capabilities."
    return rules


def bind_state_precondition(payload, *, effect, version, target):
    """Compile a known write condition, without inventing Runtime state.

    Runtime must still check the version atomically before execution. Read calls
    keep the protocol's optional explicit guard, including when no snapshot exists.
    """
    if effect != "write":
        return
    if version is None:
        from .errors import ALPError
        raise ALPError("MISSING_RUNTIME_CONTEXT", "A state-writing target requires an observed state version.",
                       422, [{"code": "MISSING_STATE_VERSION", "path": target}])
    payload["properties"]["expected_state_version"] = {"const": version}
    payload["required"] = list(dict.fromkeys([*payload.get("required", []), "expected_state_version"]))
