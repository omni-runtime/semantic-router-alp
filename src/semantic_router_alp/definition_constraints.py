"""Compile ALP definition relationships when their operands are known."""

from __future__ import annotations

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
    allowed_tools = [t for t in (list(catalog.tools) if tools is None else tools)
                     if contracts.resource_tools.get(t, {}).get("delegatable", True)]
    execution = definitions["AgentExecution"]
    execution["properties"]["tool_subset"]["items"] = (
        {"enum": allowed_tools} if allowed_tools else False
    )
    knowledge = [t for t in contracts.resource_tools if t.startswith("knowledge.")]
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
        # Resource/tool correlation is checked incrementally by the native
        # dependency matcher and independently against the final document.
        # Keep one symbolic domain instead of enumerating 2**len(resources).
        definitions["ResourceSubset"]["properties"]["slots"].update(
            maxItems=len(slots), uniqueItems=True, **{"x-alp-prefix-unique": True})
        execution["properties"]["tool_subset"]["x-alp-prefix-unique"] = True
        definitions["AgentExecution"] = execution
        declared = body["properties"]["payload"].get("properties", {}).get("requested_tools", {})
        if declared.get("type") == "array":
            needs_knowledge = any(r["kind"] == "knowledge" for r in resources)
            candidates = [t for t in catalog.tools if (t not in knowledge or needs_knowledge)
                          and contracts.resource_tools.get(t, {}).get("delegatable", True)]
            needed = list(knowledge) if needs_knowledge else []
            for resource in resources:
                if resource["kind"] == "mcp":
                    needed.extend(resource["tool_allowlist"])
            needed = list(dict.fromkeys(needed))
            if not set(needed).issubset(candidates):
                raise ALPError("INVALID_TASK_CONSTRAINT", "Resource tools are unavailable.", 422)
            declared["items"] = {"enum": candidates} if candidates else False
            declared["x-alp-required-items"] = needed
            declared["x-alp-prefix-unique"] = True
    if tools is not None and resources is not None:
        knowledge = {t for t in contracts.resource_tools if t.startswith("knowledge.")}
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
    # Do not offer declarations whose mandatory tools are unavailable: such a
    # prefix would dead-end when dependency decoding closes the resources array.
    unavailable = set()
    if not {"knowledge.search", "knowledge.read"}.issubset(allowed_tools):
        unavailable.add("knowledge")
    if not mcp_tools:
        unavailable.add("mcp")
    resource_union = definitions["ResourceRequirement"]
    if isinstance(resource_union, dict):
        names = {"KnowledgeRequirement": "knowledge", "MCPRequirement": "mcp"}
        for union in ("oneOf", "anyOf"):
            if union in resource_union:
                branches = [branch for branch in resource_union[union]
                            if names.get(branch.get("$ref", "").rsplit("/", 1)[-1],
                                         branch.get("properties", {}).get("kind", {}).get("const")) not in unavailable]
                if branches:
                    resource_union[union] = branches
                else:
                    definitions["ResourceRequirement"] = False
                    resource_array = body["properties"]["payload"]["properties"]["resource_requirements"]
                    if "const" not in resource_array:
                        resource_array.update(items=False, maxItems=0)
    # Correlate tool subsets, effects and registered handler contracts before
    # sampling; the model never supplies the trusted tool classification.
    capability = definitions["CapabilityDefinition"]
    branches = capability_branches(capability, definitions, catalog, contracts, fixed)
    definitions["CapabilityDefinition"] = {"anyOf": branches}
    bind_capabilities(body, constraints, _DependencyValidator)
    payload = body["properties"]["payload"]
    tools_schema = payload.get("properties", {}).get("requested_tools", {})
    if tools_schema.get("type") == "array" and {"knowledge.search", "knowledge.read"}.issubset(catalog.tools):
        tools_schema["x-alp-together"] = [["knowledge.search", "knowledge.read"]]
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
    # A completed resource selection narrows only that capability's later tools.
    # Prefix items keep earlier capabilities resumable without enumerating 2**N
    # resource subsets. Arrays and object order in the emitted prefix are retained.
    selections = fixed.get('/capability_resources', {})
    array = payload.get('properties', {}).get('exported_capabilities', {})
    if selections and 'const' not in array:
        import copy
        prefix = []
        generic = definitions['CapabilityDefinition']
        by_slot = {r['slot']: r for r in resources or []}
        owners = {t: {r['slot'] for r in resources or [] if r['kind'] == 'mcp' and t in r['tool_allowlist']}
                  for r in resources or [] if r['kind'] == 'mcp' for t in r['tool_allowlist']}
        for index in range(max(map(int, selections)) + 1):
            selected = selections.get(str(index))
            if selected is None:
                prefix.append({'$ref': '#/$defs/CapabilityDefinition'})
                continue
            required = {'knowledge.search', 'knowledge.read'} if any(by_slot[s]['kind'] == 'knowledge' for s in selected) else set()
            branches = []
            for original in generic.get('anyOf', [generic]):
                branch = copy.deepcopy(original)
                run = branch.get('properties', {}).get('execution', {})
                if 'tool_subset' not in run.get('properties', {}):
                    continue
                subset = run['properties']['tool_subset']
                domain = subset.get('items', {})
                allowed = domain.get('enum', []) if isinstance(domain, dict) else []
                allowed = [t for t in allowed if (not t.startswith('knowledge.') or required)
                           and (t not in owners or owners[t].intersection(selected))]
                needed = set(subset.get('x-alp-required-items', [])) | required
                highest = [t for t in subset.get('x-alp-any-items', []) if t in allowed]
                if not needed.issubset(allowed) or subset.get('x-alp-any-items') and not highest:
                    continue
                subset.update(items={'enum': allowed} if allowed else False, maxItems=len(allowed),
                              **{'x-alp-required-items': sorted(needed), 'x-alp-prefix-unique': True})
                if highest:
                    subset['x-alp-any-items'] = highest
                branches.append(branch)
            if not branches:
                raise ALPError('INVALID_TASK_CONSTRAINT', 'Resource selection cannot satisfy capability effects.', 422)
            prefix.append({'anyOf': branches})
        array['prefixItems'] = prefix
    return body
