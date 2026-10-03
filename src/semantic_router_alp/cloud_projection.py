"""Lossless typed Function Calling projection of the shared ALP contract.

The authoritative protocol permits operation-specific native projections. This
one retains the three canonical body fields, with payload as an object. It does
not fill, coerce, remove or repair any provider-generated field.
"""
from __future__ import annotations

import copy

from semantic_router_alp.catalog import stable_json
from semantic_router_alp.schema_tools import compact_schema, generation_schemas


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


def typed_arguments_schema(profile, operation):
    schema = copy.deepcopy(generation_schemas(
        profile.body_schemas[operation], operation,
        explicit_output=profile.catalog.explicit_definition_output,
        text_limit=profile.catalog.generation_text_limit,
        instruction_limit=profile.catalog.generation_instruction_limit,
    )[0])
    schema = compact_schema(schema, annotations=False)
    if operation in profile.catalog.payload_constraints:
        _expose_bound_types(schema)
        return schema
    return expose_native_schema(schema)
