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
    resources = properties.get("resource_requirements", {})
    if "const" not in resources:
        resources["maxItems"] = min(resources.get("maxItems", 16), 16)
    # Declare dependencies before capabilities, so the request-local matcher
    # can bind exact domains while decoding the rest of this same action.
    dependencies = ("resource_requirements", "requested_tools", "state_schema")
    payload["properties"] = {
        **{k: properties[k] for k in dependencies if k in properties},
        **{k: v for k, v in properties.items() if k not in dependencies},
    }
    properties = payload["properties"]
    payload["required"] = list(dict.fromkeys([*payload.get("required", []), *(k for k in dependencies[:2] if k in properties)]))
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


def generation_views(schema: dict, operation: str, *, codec="body", **options) -> list[dict]:
    """One generation view for rendering and grammar, including codec envelope."""
    if codec not in {"body", "tagged", "canonical"}:
        raise ValueError("Unknown generation codec")
    views = generation_schemas(schema, operation, **options)
    if codec == "canonical":
        for view in views:
            props = view["properties"]
            view["properties"] = {"protocol_version": props["protocol_version"],
                                  "request_id": props["request_id"],
                                  "operation": {"const": operation}, "payload": props["payload"]}
            view["required"] = list(view["properties"])
    return views


def simplify_native_schema(schema: dict) -> dict:
    """Inline small reference-only leaves without changing assertions or unions.

    Never unfold recursion or overwrite $ref siblings. Walk schema positions only,
    so JSON values inside const/enum/examples remain untouched.
    """
    result = copy.deepcopy(schema)
    definitions = result.get("$defs", {})
    for _ in range(8):
        changed = False
        def walk(node):
            nonlocal changed
            if not isinstance(node, dict):
                return
            ref = node.get("$ref", "")
            if set(node) == {"$ref"} and ref.startswith("#/$defs/"):
                name = ref[len("#/$defs/"):].replace("~1", "/").replace("~0", "~")
                target = definitions.get(name)
                if isinstance(target, dict):
                    pending = [target]
                    recursive = False
                    while pending:
                        child = pending.pop()
                        if isinstance(child, dict):
                            recursive |= "$ref" in child or "$dynamicRef" in child or "$id" in child
                            pending.extend(schema_children(child))
                    from .catalog import stable_json
                    if not recursive and len(stable_json(target)) <= 256:
                        node.clear()
                        node.update(copy.deepcopy(target))
                        changed = True
            for child in list(schema_children(node)):
                walk(child)
        walk(result)
        if not changed:
            break
    return compact_schema(result, annotations=False)
