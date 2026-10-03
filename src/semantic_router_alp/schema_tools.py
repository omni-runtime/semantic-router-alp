"""Schema presentation and strict, bounded generation-order alternatives."""

from __future__ import annotations

import copy

from .catalog import schema_children


def compact_schema(schema: dict, *, annotations: bool = True) -> dict:
    """Remove unreachable definitions, optionally prose, without dropping assertions."""
    result = copy.deepcopy(schema)
    annotation_keys = {
        "$schema", "$id", "title", "description", "default", "examples",
        "$comment", "deprecated", "readOnly", "writeOnly",
    }

    def clean(node):
        if not isinstance(node, dict):
            return
        for key in annotation_keys:
            node.pop(key, None)
        for child in schema_children(node):
            clean(child)

    if annotations:
        clean(result)
    definitions = result.pop("$defs", {})
    needed = set()
    pending = [result]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        ref = node.get("$ref", "")
        if ref.startswith("#/$defs/"):
            name = ref.split("/")[2].replace("~1", "/").replace("~0", "~")
            if name not in needed:
                needed.add(name)
                pending.append(definitions[name])
        pending.extend(schema_children(node))
    if needed:
        result["$defs"] = {name: value for name, value in definitions.items() if name in needed}
    return result


def generation_schemas(
    schema: dict, operation: str, *, flexible: bool = False, explicit_output: bool = False,
    text_limit: int = 0, instruction_limit: int = 0,
) -> list[dict]:
    """Allow late definition output/environment without arbitrary key repetition.

    Each branch retains all required fields, value constraints and object limits.
    Do not use XGrammar 0.2.8 any_order: it drops required-key/uniqueness checks.
    """
    result = copy.deepcopy(schema)
    if operation != "agent_definition_generate":
        return [result]
    # Only ALP prose fields, never identically named business properties. Do not
    # follow arbitrary $refs into tool arguments or embedded value schemas.
    def limit_prose(node):
        if not isinstance(node, dict):
            return
        for key, limit in (("description", text_limit), ("instructions", instruction_limit)):
            value = node.get("properties", {}).get(key, {})
            if limit and value.get("type") == "string" and not {"const", "enum"} & value.keys():
                value["maxLength"] = min(value.get("maxLength", limit), limit)
        for union in ("anyOf", "oneOf"):
            for branch in node.get(union, []):
                limit_prose(branch)
        execution = node.get("properties", {}).get("execution")
        if execution:
            limit_prose(execution)

    limit_prose(result["properties"]["payload"])
    for name in ("CapabilityDefinition", "AgentExecution", "KnowledgeRequirement",
                 "SkillRequirement", "MCPRequirement", "AssetRequirement"):
        limit_prose(result.get("$defs", {}).get(name, {}))
    payload = result["properties"]["payload"]
    if "properties" not in payload:
        return [result]
    properties = payload["properties"]
    resources = properties["resource_requirements"]
    if "const" not in resources:
        # Dependency compilation enumerates resource subsets. Keep generated
        # declarations inside that supported domain instead of dead-ending
        # only after the ninth resource has already been emitted.
        resources["maxItems"] = min(resources.get("maxItems", 8), 8)
    # Declare dependencies before capabilities, so the request-local matcher
    # can bind exact domains while decoding the rest of this same action.
    dependencies = ("resource_requirements", "requested_tools", "state_schema")
    payload["properties"] = {
        **{k: properties[k] for k in dependencies},
        **{k: v for k, v in properties.items() if k not in dependencies},
    }
    properties = payload["properties"]
    payload["required"] = list(dict.fromkeys([*payload.get("required", []), *dependencies[:2]]))
    # An operator may require an explicit output contract during sampling;
    # final ALP validation still accepts
    # legacy definitions which rely on the protocol's default output contract.
    if explicit_output:
        payload["required"] = list(dict.fromkeys([*payload.get("required", []), "output"]))
    if not flexible:
        return [result]
    late = copy.deepcopy(payload)
    tail = ("output", "environment_profile_ref")
    late["properties"] = {
        **{k: v for k, v in properties.items() if k not in tail},
        **{k: properties[k] for k in tail if k in properties},
    }
    alternative = copy.deepcopy(result)
    alternative["properties"]["payload"] = late
    return [result, alternative]
