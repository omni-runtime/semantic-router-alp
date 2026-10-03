"""Lossless typed Function Calling projection of the shared ALP contract.

The authoritative protocol permits operation-specific native projections. This
one retains the three canonical body fields, with payload as an object. It does
not fill, coerce, remove or repair any provider-generated field.
"""
from __future__ import annotations

import copy

from semantic_router_alp.catalog import stable_json
from semantic_router_alp.schema_tools import compact_schema


def cloud_body_schema(profile, operation, index=None):
    """Keep protocol/host assertions without local dependency-first decoding.

    Optional dependencies remain optional unless the caller binds them. Explicit
    operator output/prose limits are independent of the local token matcher.
    """
    schema = copy.deepcopy(profile.member_schemas[index] if index is not None
                           else profile.body_schemas[operation])
    if operation == "agent_definition_generate":
        payload = schema["properties"]["payload"]
        if profile.catalog.explicit_definition_output and "output" in payload.get("properties", {}):
            payload["required"] = list(dict.fromkeys([*payload.get("required", []), "output"]))

        def limit_prose(node):
            if not isinstance(node, dict):
                return
            for field, bound in (("description", profile.catalog.generation_text_limit),
                                 ("instructions", profile.catalog.generation_instruction_limit)):
                value = node.get("properties", {}).get(field, {})
                if bound and value.get("type") == "string" and not {"const", "enum"} & value.keys():
                    value["maxLength"] = min(value.get("maxLength", bound), bound)
            for union in ("anyOf", "oneOf"):
                for branch in node.get(union, []):
                    limit_prose(branch)
            limit_prose(node.get("properties", {}).get("execution"))

        limit_prose(payload)
        for name in ("CapabilityDefinition", "AgentExecution", "KnowledgeRequirement",
                     "SkillRequirement", "MCPRequirement", "AssetRequirement"):
            limit_prose(schema.get("$defs", {}).get(name))
    return compact_schema(schema, annotations=False)


def describe_cloud_fields(schema, operation):
    """Annotate ALP field positions only, never user-owned schema literals."""
    from .semantics import OPERATION_DESCRIPTIONS

    payload = schema.get("properties", {}).get("payload")
    if payload is None:
        return schema
    payload["description"] = OPERATION_DESCRIPTIONS[operation]
    if operation != "agent_definition_generate":
        return schema
    descriptions = {
        "name": "Agent name. Copy the name explicitly supplied by the caller exactly; choose a name only when none is supplied. Put explanatory wording in description, not in a renamed identifier.",
        "input_schema": "Input contract for the whole agent_call.input.arguments. Separate from every exported capability input_schema and from task text. Omission means an empty-object-only interface. Do not copy a capability's input here unless the task also asks for that whole-agent interface.",
        "output": "Final output contract for the whole Agent run. When the task requests the same JSON contract as a capability output_schema, use that same schema, not an example value.",
        "requested_tools": "Requested tool references, not an execution schedule. Preserve the caller's explicitly specified list order. Tool availability alone does not request all visible tools or create resource dependencies. Omission defaults to an empty list.",
        "resource_requirements": "Separately requested knowledge/skill/mcp/asset slots. A requested tool does not automatically require a new resource slot or profile. For exact encoding, omit this field or use [] when no resource dependencies are supplied. Omission defaults to [].",
        "exported_capabilities": "Only the capabilities requested by the task. Every capability requires name, description, input_schema, output_schema, execution, state_effect and external_effect. Preserve the specified interface fields; do not add extra result fields or methods.",
    }

    def annotate_payload(node):
        for name, description in descriptions.items():
            field = node.get("properties", {}).get(name)
            if isinstance(field, dict):
                field["description"] = description
        for union in ("anyOf", "oneOf", "allOf"):
            for branch in node.get(union, []):
                if isinstance(branch, dict):
                    annotate_payload(branch)

    annotate_payload(payload)
    capability = schema.get("$defs", {}).get("CapabilityDefinition", {}).get("properties", {})
    for name, description in {
        "input_schema": "Arguments accepted by this named capability only. This does not define the whole Agent input interface.",
        "output_schema": "Result returned by this named capability. Reuse its schema for whole-Agent output only when the task explicitly requests that relationship.",
        "state_effect": "Required business-state effect: read means no business-state mutation, including when there is no state access; write means business-state mutation. This is separate from external_effect.",
        "external_effect": "Required external effect: none for no external access, read for external reads, write for external mutation. It must cover every selected tool's declared effect; using a read tool cannot declare none.",
    }.items():
        if isinstance(capability.get(name), dict):
            capability[name]["description"] = description
    return schema


def native_scalar_enums(schema):
    """Use equivalent singleton enums for provider-native scalar constants.

    Visit schema positions only: fixed object values and embedded literal schema
    documents remain opaque. Canonical validation still checks the original ALP.
    """
    def walk(node):
        if not isinstance(node, dict):
            return
        if "const" in node and "enum" not in node and not isinstance(node["const"], (dict, list)):
            node["enum"] = [node.pop("const")]
        for child in _schema_children(node):
            walk(child)
    walk(schema)
    return schema


def _schema_children(node):
    """Visit schema positions, never literal host values in const/enum/default."""
    for key in ("properties", "$defs", "patternProperties", "dependentSchemas"):
        yield from node.get(key, {}).values()
    for key in ("items", "additionalProperties", "not", "if", "then", "else", "contains",
                "propertyNames", "unevaluatedProperties", "unevaluatedItems"):
        if key in node:
            yield node[key]
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        yield from node.get(key, [])


def _expose_bound_types(node):
    """Keep fixed host contracts compact instead of duplicating union properties.

    Bound values already provide the model's task requirements. Retain the
    original assertions and only expose direct object-union/string-constant types.
    """
    if not isinstance(node, dict):
        return
    for child in _schema_children(node):
        _expose_bound_types(child)
    branches = node.get("anyOf", node.get("oneOf", []))
    if branches and all(isinstance(b, dict) and b.get("type") == "object" for b in branches):
        node.setdefault("type", "object")
    if isinstance(node.get("const"), str):
        node.setdefault("type", "string")


def expose_native_schema(schema):
    """Expose redundant type/property information for native tool renderers.

    Keep every original reference and union assertion. The added assertions are
    already entailed by them; this neither relaxes the protocol nor repairs a
    generated response. Local references are inspected without external I/O.
    """
    definitions = schema.get("$defs", {})

    def inferred_type(node, seen=frozenset()):
        if not isinstance(node, dict):
            return None
        if isinstance(node.get("type"), str):
            return node["type"]
        if "const" in node:
            value = node["const"]
            if value is None:
                return "null"
            for python_type, json_type in ((bool, "boolean"), (dict, "object"),
                                           (list, "array"), (str, "string"),
                                           (int, "number"), (float, "number")):
                if isinstance(value, python_type):
                    return json_type
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/") and ref not in seen:
            name = ref[8:]
            if "/" not in name:
                target = definitions.get(name.replace("~1", "/").replace("~0", "~"))
                kind = inferred_type(target, seen | {ref})
                if kind:
                    return kind
        for key in ("anyOf", "oneOf"):
            if node.get(key):
                kinds = {inferred_type(child, seen) for child in node[key]}
                if len(kinds) == 1 and None not in kinds:
                    return next(iter(kinds))
        kinds = {inferred_type(child, seen) for child in node.get("allOf", [])} - {None}
        return next(iter(kinds)) if len(kinds) == 1 else None

    def walk(node):
        if not isinstance(node, dict):
            return
        kind = inferred_type(node)
        if kind:
            node.setdefault("type", kind)
        for child in _schema_children(node):
            walk(child)
        branches = node.get("anyOf", node.get("oneOf", []))
        # Restrict expansion to explicitly closed object branches. A branch with
        # patternProperties can admit names absent from properties, so it must
        # not acquire a closed parent. Existing sibling constraints stay intact.
        if (not branches or any(k in node for k in ("properties", "patternProperties", "additionalProperties"))
                or not all(isinstance(b, dict) and b.get("type") == "object"
                           and b.get("additionalProperties") is False and not b.get("patternProperties")
                           for b in branches)):
            return
        properties = {}
        for name in dict.fromkeys(k for b in branches for k in b.get("properties", {})):
            variants = {stable_json(b["properties"][name]): b["properties"][name]
                        for b in branches if name in b.get("properties", {})}
            values = list(variants.values())
            if len(values) == 1:
                value = copy.deepcopy(values[0])
            elif all(isinstance(v, dict) and set(v) <= {"const", "type"}
                     and isinstance(v.get("const"), str) and v.get("type", "string") == "string" for v in values):
                value = {"type": "string", "enum": [v["const"] for v in values]}
            else:
                value = {"anyOf": copy.deepcopy(values)}
                walk(value)
            properties[name] = value
        node["properties"] = properties
        node["additionalProperties"] = False
        common = set.intersection(*(set(b.get("required", [])) for b in branches))
        node.setdefault("required", [name for name in properties if name in common])

    walk(schema)
    return schema


def typed_arguments_schema(profile, operation, index=None):
    from .schema_tools import simplify_native_schema
    schema = simplify_native_schema(_typed_arguments_schema(profile, operation, index))
    if profile.catalog.semantic_rendering:
        describe_cloud_fields(schema, operation)
    return native_scalar_enums(schema)


def _typed_arguments_schema(profile, operation, index=None):
    if profile.member_schemas and index is None:
        from .constraints import _import_schema
        candidates = [typed_arguments_schema(profile, operation, i)
                      for i, member in enumerate(profile.catalog.response_constraints.members)
                      if member.operation == operation]
        if len(candidates) == 1:
            return candidates[0]
        # Most ordered calls share the same operation/catalog definitions.
        # Factor those assertions once instead of importing every schema tree.
        # This is an equivalent intersection of common assertions and a union
        # of member payloads, not a widened interface.
        common = candidates[0]
        def comparable(item):
            return {k: v for k, v in item.items() if k != "properties"}
        if (all(stable_json(comparable(c)) == stable_json(comparable(common)) for c in candidates)
                and all(stable_json({k: v for k, v in c["properties"].items() if k != "payload"}) ==
                        stable_json({k: v for k, v in common["properties"].items() if k != "payload"}) for c in candidates)):
            root = copy.deepcopy(common)
            payloads = [copy.deepcopy(c["properties"]["payload"]) for c in candidates]
            shared = payloads[0].get("allOf", [])
            same_shared = all(stable_json(p.get("allOf", [])) == stable_json(shared) for p in payloads)
            if same_shared:
                for payload in payloads:
                    payload.pop("allOf", None)
            root["properties"]["payload"] = {"anyOf": payloads}
            if same_shared and shared:
                root["properties"]["payload"]["allOf"] = shared
            return expose_native_schema(root)
        root = {"anyOf": []}
        for candidate in candidates:
            reference = _import_schema(root, candidate)
            imported = root["$defs"][reference["$ref"].rsplit("/", 1)[-1]]
            # Expose the body at the root; provider renderers often ignore a
            # root consisting only of referenced anyOf branches.
            root["anyOf"].append({k: copy.deepcopy(v) for k, v in imported.items() if k != "$defs"})
        return expose_native_schema(root)
    schema = cloud_body_schema(profile, operation, index)
    if profile.member_schemas or operation in profile.catalog.payload_constraints:
        _expose_bound_types(schema)
        return schema
    return expose_native_schema(schema)
