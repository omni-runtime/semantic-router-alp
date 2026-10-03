"""Compile ALP definition relationships when their operands are known."""

from __future__ import annotations

import copy
import itertools

from jsonschema import Draft202012Validator, ValidationError, validators

from .errors import ALPError
from .host_contracts import bind_capabilities, capability_branches, narrow_resources


def _required_items(validator, required, instance, schema):
    if isinstance(instance, list) and any(value not in instance for value in required):
        yield ValidationError("Missing required dependency item")


def _together(validator, groups, instance, schema):
    if isinstance(instance, list):
        for group in groups:
            if any(value in instance for value in group) and not all(value in instance for value in group):
                yield ValidationError("Incomplete dependency group")


def _any_items(validator, choices, instance, schema):
    if isinstance(instance, list) and not any(value in instance for value in choices):
        yield ValidationError("Missing effect-class item")


_DependencyValidator = validators.extend(Draft202012Validator, {
    "x-alp-required-items": _required_items, "x-alp-together": _together,
    "x-alp-any-items": _any_items,
})


def specialize_definition(body, catalog, contracts):
    definitions = body["$defs"]
    constraints = catalog.payload_constraints.get("agent_definition_generate")
    fixed = constraints.fixed_values if constraints else {}
    tools = fixed.get("/requested_tools")
    resources = fixed.get("/resource_requirements")
    allowed_tools = list(catalog.tools) if tools is None else tools
    execution = definitions["AgentExecution"]
    execution["properties"]["tool_subset"]["items"] = (
        {"enum": allowed_tools} if allowed_tools else False
    )
    knowledge = list(contracts.resource_tools)
    # An agent_run using either knowledge tool necessarily selects a knowledge
    # resource, which requires both tools. This implication does not depend on
    # the resource slot names being known yet.
    if set(knowledge).issubset(allowed_tools):
        execution["properties"]["tool_subset"]["x-alp-together"] = [knowledge]
    else:
        allowed_tools = [t for t in allowed_tools if t not in knowledge]
        execution["properties"]["tool_subset"]["items"] = (
            {"enum": allowed_tools} if allowed_tools else False
        )
    mcp_tools = [t for t in allowed_tools if not t.startswith("knowledge.")]
    definitions["MCPRequirement"]["properties"]["tool_allowlist"]["items"] = (
        {"enum": mcp_tools} if mcp_tools else False
    )
    if not mcp_tools:
        resource_union = definitions["ResourceRequirement"]
        for union in ("oneOf", "anyOf"):
            if union in resource_union:
                resource_union[union] = [branch for branch in resource_union[union]
                                         if branch != {"$ref": "#/$defs/MCPRequirement"}]
    if resources is not None:
        slots = [r["slot"] for r in resources]
        if len(set(slots)) != len(slots):
            raise ALPError("INVALID_TASK_CONSTRAINT", "Resource slots must be unique.", 422)
        definitions["ResourceSubset"]["properties"]["slots"]["items"] = (
            {"enum": slots} if slots else False
        )
        if len(slots) > 8:
            raise ALPError("UNSUPPORTED_SCHEMA", "Strict capability dependencies support at most 8 resource slots.", 422)
        # Select a resource subset and compile its exact tool dependencies in
        # the same branch. Keep arbitrary array order within that subset.
        executions = []
        for flags in itertools.product((False, True), repeat=len(resources)):
            selected = [r for r, flag in zip(resources, flags, strict=True) if flag]
            selected_slots = [r["slot"] for r in selected]
            knowledge_selected = any(r["kind"] == "knowledge" for r in selected)
            mcp_owners = {t: r["slot"] for r in resources if r["kind"] == "mcp" for t in r["tool_allowlist"]}
            possible = [t for t in allowed_tools if
                        (t not in knowledge or knowledge_selected) and
                        (t not in mcp_owners or mcp_owners[t] in selected_slots)]
            if knowledge_selected and not set(knowledge).issubset(possible):
                continue
            branch = copy.deepcopy(execution)
            branch["properties"]["tool_subset"] = {
                "type": "array", "items": {"enum": possible} if possible else False,
                "uniqueItems": True, "maxItems": len(possible),
                "x-alp-required-items": knowledge if knowledge_selected else [],
            }
            branch["properties"]["resource_subset"] = {
                "type": "object", "properties": {"slots": {
                    "type": "array", "items": {"enum": selected_slots} if selected_slots else False,
                    "minItems": len(selected_slots), "maxItems": len(selected_slots), "uniqueItems": True,
                }}, "required": ["slots"], "additionalProperties": False,
            }
            if selected_slots:
                branch["required"] = list(dict.fromkeys([*branch["required"], "resource_subset"]))
            executions.append(branch)
        definitions["AgentExecution"] = {"anyOf": executions}
        declared = body["properties"]["payload"].get("properties", {}).get("requested_tools", {})
        if declared.get("type") == "array":
            needs_knowledge = any(r["kind"] == "knowledge" for r in resources)
            candidates = [t for t in catalog.tools if t not in knowledge or needs_knowledge]
            needed = list(knowledge) if needs_knowledge else []
            for resource in resources:
                if resource["kind"] == "mcp":
                    needed.extend(resource["tool_allowlist"])
            needed = list(dict.fromkeys(needed))
            if not set(needed).issubset(candidates):
                raise ALPError("INVALID_TASK_CONSTRAINT", "Resource tools are unavailable.", 422)
            declared["items"] = {"enum": candidates} if candidates else False
            declared["x-alp-required-items"] = needed
    if tools is not None and resources is not None:
        knowledge = set(contracts.resource_tools)
        has_knowledge = any(r["kind"] == "knowledge" for r in resources)
        required = knowledge if has_knowledge else set()
        for resource in resources:
            if resource["kind"] == "mcp":
                required = required | set(resource["tool_allowlist"])
        if not required.issubset(tools) or (set(tools) & knowledge and not has_knowledge):
            raise ALPError(
                "INVALID_TASK_CONSTRAINT", "Fixed tools and resources violate ALP dependencies.", 422
            )
    narrow_resources(body, catalog)
    # Correlate tool subsets, effects and registered handler contracts before
    # sampling; the model never supplies the trusted tool classification.
    capability = definitions["CapabilityDefinition"]
    branches = capability_branches(capability, definitions, catalog, contracts, fixed)
    definitions["CapabilityDefinition"] = {"anyOf": branches}
    bind_capabilities(body, constraints, _DependencyValidator)
    payload = body["properties"]["payload"]
    tools_schema = payload.get("properties", {}).get("requested_tools", {})
    if tools_schema.get("type") == "array" and set(contracts.resource_tools).issubset(catalog.tools):
        tools_schema["x-alp-together"] = [list(contracts.resource_tools)]
    state_schema = fixed.get("/state_schema")
    if state_schema is not None:
        # state_schema precedes initial_state in the contract. Once emitted,
        # its values become the actual schema for the later state object.
        from .constraints import _import_schema
        payload["properties"]["initial_state"] = _import_schema(body, state_schema)
        if not Draft202012Validator(state_schema).is_valid({}):
            payload["required"] = list(dict.fromkeys([*payload["required"], "initial_state"]))
        if "/initial_state" in fixed and not Draft202012Validator(state_schema).is_valid(fixed["/initial_state"]):
            raise ALPError("INVALID_TASK_CONSTRAINT", "Fixed initial state violates its state schema.", 422)
    # Revalidate fixed complete fields against the specialized relationships.
    properties = contracts.protocol_schema["$defs"]["DefinitionDraft"]["properties"]
    for path, value in fixed.items():
        if path.count("/") == 1 and path[1:] in properties:
            schema = {"$defs": definitions, **properties[path[1:]]}
            if not _DependencyValidator(schema).is_valid(value):
                raise ALPError(
                    "INVALID_TASK_CONSTRAINT", "A fixed definition field violates its dependencies.",
                    422, [{"path": "/payload" + path}],
                )
    return body
