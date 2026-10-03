"""Compile and independently validate request-scoped, trusted host contracts."""
from __future__ import annotations

import copy

from jsonschema import Draft202012Validator

from .catalog import stable_json
from .errors import ALPError

EMPTY_STATE = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
EFFECTS = ("none", "read", "write")


def task_constraint_coverage(canonical, catalog):
    """Describe enforced host requirements without echoing values or claiming NL coverage."""
    operation = canonical["operation"]
    constraints = catalog.payload_constraints.get(operation)
    fixed = constraints.fixed_values if constraints else {}
    required = constraints.required_fields if constraints else []

    def mode(path):
        for parent in sorted(fixed, key=len, reverse=True):
            if path == parent:
                return "fixed"
            if path.startswith(parent + "/"):
                value = fixed[parent]
                try:
                    for part in path[len(parent) + 1:].split("/"):
                        value = value[part.replace("~1", "/").replace("~0", "~")]
                except (KeyError, TypeError):
                    return "fixed_absent"
                return "fixed"
        if path in required or any(p.startswith(path + "/") for p in [*required, *fixed]):
            return "required_model_value"
        if operation == "agent_call" and path == "/session_mode" and catalog.explicit_session_mode:
            return "required_model_value"
        return "not_task_bound"

    paths = {
        "agent_call": ["/instance_id", "/input/task", "/input/artifacts", "/session_mode"],
        "agent_definition_generate": ["/name", "/output", "/environment_profile_ref"],
    }.get(operation, [])
    capabilities = {}
    if operation == "agent_definition_generate":
        for index, capability in enumerate(canonical["payload"].get("exported_capabilities", [])):
            contract = constraints.capabilities.get(capability["name"]) if constraints else None
            capabilities[capability["name"]] = {
                key: "fixed" if mode("/exported_capabilities") == "fixed" or (
                    contract is not None and getattr(contract, key) is not None
                ) else "not_task_bound"
                for key in ("input_schema", "output_schema", "state_effect", "external_effect")
            }
    return {
        "fixed_paths": sorted(fixed),
        "required_paths": sorted(required),
        "key_fields": {path: mode(path) for path in paths},
        "capability_contracts": capabilities,
        "explicit_session_mode": operation == "agent_call" and catalog.explicit_session_mode,
        "natural_language_requirements": "not_evaluated",
    }


def validation_scope(canonical, catalog):
    operation = canonical["operation"]
    definition = operation == "agent_definition_generate"
    tool = canonical["payload"].get("tool") if operation == "tool_call" else None
    relevant = {
        "tool_effects": definition,
        "environment_profiles": definition,
        "resource_bindings": definition or tool in {"knowledge.search", "knowledge.read", "skill.activate", "asset.read"},
        "evidence_bindings": tool == "knowledge.read",
    }
    result = {"protocol_static": "passed", "task_constraints":
              "passed" if operation in catalog.payload_constraints else "not_provided"}
    for field, applies in relevant.items():
        provided = field == "tool_effects" or getattr(catalog, field) is not None
        result[field] = "not_applicable" if not applies else "passed" if provided else "not_provided"
    return result


def tool_effects(catalog, contracts):
    effects = {}
    for name, tool in catalog.tools.items():
        reserved = contracts.resource_tools.get(name)
        effect = reserved["external_effect"] if reserved else tool.external_effect
        if effect is None:
            raise ALPError("UNCLASSIFIED_TOOL", "Definition generation requires trusted tool effects.", 422)
        effects[name] = effect
    return effects


def narrow_resources(body, catalog):
    definitions = body["$defs"]
    payload = body["properties"]["payload"]
    if catalog.environment_profiles is not None:
        payload["properties"]["environment_profile_ref"] = (
            {"enum": catalog.environment_profiles} if catalog.environment_profiles else False
        )
        if "default" not in catalog.environment_profiles:
            payload["required"] = list(dict.fromkeys([*payload["required"], "environment_profile_ref"]))
    if catalog.resource_bindings is None:
        return
    names = {"knowledge": "KnowledgeRequirement", "skill": "SkillRequirement",
             "mcp": "MCPRequirement", "asset": "AssetRequirement"}
    branches = []
    for binding in catalog.resource_bindings:
        branch = copy.deepcopy(definitions[names[binding.kind]])
        for key, value in binding.model_dump(exclude_none=True).items():
            # A host binding is an exact, correlated tuple, not independent enums.
            branch["properties"][key] = {"const": value}
        branches.append(branch)
    definitions["ResourceRequirement"] = {"anyOf": branches} if branches else False


def tool_argument_variants(name, tool, catalog):
    schema = tool.input_schema
    if catalog.resource_bindings is None:
        return [schema]
    kind = {"knowledge.search": "knowledge", "knowledge.read": "knowledge",
            "skill.activate": "skill", "asset.read": "asset"}.get(name)
    if kind is None:
        return [schema]
    bindings = [{"slot": b.slot} for b in catalog.resource_bindings if b.kind == kind]
    if name == "knowledge.read" and catalog.evidence_bindings is not None:
        bindings = [{"slot": slot, "evidence_id": evidence}
                    for evidence, slot in catalog.evidence_bindings.items()]
    result = []
    for fixed in bindings:
        branch = copy.deepcopy(schema)
        for key, value in fixed.items():
            field = branch.get("properties", {}).get(key)
            if field is None or not Draft202012Validator({"$defs": schema.get("$defs", {}), **field}).is_valid(value):
                raise ALPError("INVALID_HOST_CATALOG", "A binding conflicts with a tool input contract.", 422)
            branch["properties"][key] = {"const": value}
            branch["required"] = list(dict.fromkeys([*branch.get("required", []), key]))
        result.append(branch)
    return result


def capability_branches(base, definitions, catalog, contracts, fixed):
    effects = tool_effects(catalog, contracts)
    agent = definitions["AgentExecution"]
    executions = agent.get("anyOf", [agent])
    branches = []
    for execution in executions:
        source = execution["properties"]["tool_subset"]
        possible = source.get("items", {})
        possible = possible.get("enum", []) if isinstance(possible, dict) else []
        possible = [t for t in possible if catalog.tools[t].state_effect == "read"]
        for level, effect in enumerate(EFFECTS):
            allowed = [t for t in possible if EFFECTS.index(effects[t]) <= level]
            highest = [t for t in allowed if effects[t] == effect]
            required = source.get("x-alp-required-items", [])
            if not set(required).issubset(allowed) or (level and not highest):
                continue
            branch = copy.deepcopy(base)
            run = copy.deepcopy(execution)
            subset = run["properties"]["tool_subset"]
            subset.update(items={"enum": allowed} if allowed else False,
                          uniqueItems=True, maxItems=len(allowed))
            if level:
                subset["x-alp-any-items"] = highest
            branch["properties"].update(execution=run, state_effect={"const": "read"},
                                         external_effect={"const": effect})
            branches.append(branch)
    tools = fixed.get("/requested_tools")
    state = fixed.get("/state_schema")
    for ref, handler in catalog.handlers.items():
        if tools is not None and not set(handler.required_tools).issubset(tools):
            continue
        if state is not None and stable_json(state) != stable_json(handler.state_schema):
            continue
        branch = copy.deepcopy(base)
        run = copy.deepcopy(definitions["HandlerExecution"])
        run["properties"]["handler_ref"] = {"const": ref}
        branch["properties"]["execution"] = run
        for key in ("input_schema", "output_schema", "state_effect", "external_effect"):
            branch["properties"][key] = {"const": getattr(handler, key)}
        branches.append(branch)
    return branches


def bind_capabilities(body, constraints, validator_class):
    if not constraints or not constraints.capabilities:
        return
    definitions = body["$defs"]
    branches = []
    for name, contract in constraints.capabilities.items():
        start = len(branches)
        for source in definitions["CapabilityDefinition"].get("anyOf", []):
            branch = copy.deepcopy(source)
            fields = {"name": name, **contract.model_dump(exclude_none=True)}
            if any(not validator_class({"$defs": definitions, **branch["properties"][key]}).is_valid(value)
                   for key, value in fields.items()):
                continue
            for key, value in fields.items():
                branch["properties"][key] = {"const": value}
            branches.append(branch)
        if len(branches) == start:
            raise ALPError("INVALID_TASK_CONSTRAINT", "A named capability has no compatible host contract.", 422)
    if not branches:
        raise ALPError("INVALID_TASK_CONSTRAINT", "Capability requirements conflict with host contracts.", 422)
    definitions["CapabilityDefinition"] = {"anyOf": branches}
    array = body["properties"]["payload"]["properties"]["exported_capabilities"]
    if "const" not in array:
        array.update(minItems=len(constraints.capabilities), maxItems=len(constraints.capabilities))
    required = body["properties"]["payload"]["required"]
    if "exported_capabilities" not in required:
        required.append("exported_capabilities")


def validate_host_contracts(canonical, catalog, contracts):
    """Independent defense: use original host data, not transformed schemas."""
    payload, operation = canonical["payload"], canonical["operation"]
    errors = []

    def fail(path, code="HOST_CONTRACT_MISMATCH"):
        errors.append({"code": code, "path": "/payload" + path})

    constraints = catalog.payload_constraints.get(operation)
    if constraints:
        for path in set(constraints.required_fields) | constraints.fixed_values.keys():
            value = payload
            try:
                for part in path[1:].split("/"):
                    value = value[part.replace("~1", "/").replace("~0", "~")]
                if path in constraints.fixed_values and stable_json(value) != stable_json(constraints.fixed_values[path]):
                    fail(path, "HOST_FIXED_VALUE_MISMATCH")
            except (KeyError, TypeError):
                fail(path, "HOST_REQUIRED_FIELD_MISSING")
        if constraints.capabilities:
            caps = payload.get("exported_capabilities", [])
            if len(caps) != len(constraints.capabilities) or {c["name"] for c in caps} != set(constraints.capabilities):
                fail("/exported_capabilities")
            for index, cap in enumerate(caps):
                contract = constraints.capabilities.get(cap["name"])
                if contract:
                    for key, value in contract.model_dump(exclude_none=True).items():
                        if stable_json(cap[key]) != stable_json(value):
                            fail(f"/exported_capabilities/{index}/{key}")
    if operation == "tool_call":
        variants = tool_argument_variants(payload["tool"], catalog.tools[payload["tool"]], catalog)
        if not any(Draft202012Validator(s).is_valid(payload["arguments"]) for s in variants):
            fail("/arguments")
    if operation != "agent_definition_generate":
        return errors
    if catalog.environment_profiles is not None and payload.get("environment_profile_ref", "default") not in catalog.environment_profiles:
        fail("/environment_profile_ref")
    if catalog.resource_bindings is not None:
        bindings = {b.slot: b for b in catalog.resource_bindings}
        for index, resource in enumerate(payload.get("resource_requirements", [])):
            binding = bindings.get(resource["slot"])
            if binding is None or any(stable_json(resource.get(k)) != stable_json(v) for k, v in binding.model_dump(exclude_none=True).items()):
                fail(f"/resource_requirements/{index}")
    effects = tool_effects(catalog, contracts)
    for index, cap in enumerate(payload.get("exported_capabilities", [])):
        execution = cap["execution"]
        path = f"/exported_capabilities/{index}"
        if execution["kind"] == "agent_run":
            selected = execution.get("tool_subset", [])
            if any(t not in effects or catalog.tools[t].state_effect != "read" for t in selected):
                fail(path + "/execution/tool_subset")
            else:
                effect = max((effects[t] for t in selected), key=EFFECTS.index, default="none")
                if cap["external_effect"] != effect:
                    fail(path + "/external_effect")
        else:
            handler = catalog.handlers.get(execution["handler_ref"])
            if handler is None:
                fail(path + "/execution/handler_ref")
                continue
            for key in ("input_schema", "output_schema", "state_effect", "external_effect"):
                if stable_json(cap[key]) != stable_json(getattr(handler, key)):
                    fail(path + "/" + key)
            if stable_json(payload.get("state_schema", EMPTY_STATE)) != stable_json(handler.state_schema):
                fail("/state_schema")
            if not set(handler.required_tools).issubset(payload.get("requested_tools", [])):
                fail(path + "/execution/handler_ref")
    return errors
