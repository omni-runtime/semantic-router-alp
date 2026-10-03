"""Narrow a catalog schema using trusted, server-owned task requirements."""

from __future__ import annotations

import copy

from jsonschema import Draft202012Validator

from .catalog import PayloadConstraints
from .errors import ALPError

_UNSET = object()


def constrain_payload(body: dict, constraints: PayloadConstraints) -> dict:
    result = copy.deepcopy(body)

    def reject(path):
        return ALPError(
            "INVALID_TASK_CONSTRAINT", "A server task constraint conflicts with this catalog.",
            422, [{"path": "/payload" + path}],
        )

    def resolve(node, path):
        if not isinstance(node, dict):
            raise reject(path)
        seen = set()
        while "$ref" in node:
            ref = node["$ref"]
            if ref in seen or set(node) - {"$ref", "description", "title"}:
                raise reject(path)
            seen.add(ref)
            target = result
            for part in ref[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
            node = copy.deepcopy(target)
        return node

    def narrow(node, parts, path, value=_UNSET):
        node = copy.deepcopy(node)
        if not parts:
            if value is _UNSET:
                return node
            schema = node if isinstance(node, bool) else {"$defs": result.get("$defs", {}), **node}
            if not Draft202012Validator(schema).is_valid(value):
                raise reject(path)
            # A verified singleton is a subset of the original leaf schema.
            return {"const": copy.deepcopy(value)}
        node = resolve(node, path)
        for union in ("anyOf", "oneOf"):
            if union in node:
                branches = []
                for branch in node[union]:
                    try:
                        branches.append(narrow(branch, parts, path, value))
                    except ALPError:
                        continue
                if not branches:
                    raise reject(path)
                node[union] = branches
                return node
        name, *rest = parts
        if node.get("type") != "object" or name not in node.get("properties", {}):
            raise reject(path)
        node["properties"][name] = narrow(node["properties"][name], rest, path, value)
        node["required"] = list(dict.fromkeys([*node.get("required", []), name]))
        return node

    original = copy.deepcopy(result["properties"]["payload"])
    payload = result["properties"]["payload"]
    for path in constraints.required_fields:
        parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
        payload = narrow(payload, parts, path)
    # Apply children before parents so a fixed object must satisfy child constraints too.
    for path, value in sorted(constraints.fixed_values.items(), key=lambda pair: -pair[0].count("/")):
        parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
        payload = narrow(payload, parts, path, value)
    # Retain the original catalog assertion as well. In particular, narrowing
    # oneOf branches must not make an originally ambiguous value become valid.
    payload["allOf"] = [*payload.get("allOf", []), original]
    result["properties"]["payload"] = payload
    return result
