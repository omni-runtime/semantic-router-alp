"""Protocol specialization without tokenizers, inference engines or grammar backends."""
from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

from alp_schema_mcp.catalog import OPERATION_DEFS, ContractCatalog
from jsonschema import Draft202012Validator

from .catalog import Catalog, schema_children, stable_json
from .errors import ALPError
from .protocol import ALPOptions, OperationChoice
from .task_constraints import constrain_payload


def _import_schema(body: dict, schema: dict) -> dict:
    """Place a dynamic schema in its own local-reference namespace."""
    definitions = body.setdefault("$defs", {})
    name = f"alp_dynamic_{len(definitions)}"
    while name in definitions:
        name += "_"
    imported = copy.deepcopy(schema)
    pending = [imported]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        if "$ref" in node:
            node["$ref"] = f"#/$defs/{name}" + node["$ref"][1:]
        pending.extend(schema_children(node))
    definitions[name] = imported
    return {"$ref": f"#/$defs/{name}"}



@dataclass(frozen=True)
class CompiledProfile:
    digest: str
    operations: tuple[str, ...]
    body_schemas: dict[str, dict]
    grammar: str
    residual_checks: tuple[dict[str, str], ...]
    catalog: Catalog



class ContractCompiler:
    compiler_identity = "semantic-router-alp/native-contract-1"

    def __init__(
        self,
        contracts: ContractCatalog | None = None,
        cache_size: int = 64,
        observed_definition: dict | None = None,
    ):
        self.contracts = contracts or ContractCatalog()
        self.cache_size = cache_size
        self._cache: OrderedDict[str, CompiledProfile] = OrderedDict()
        self._lock = threading.Lock()
        self.observed_definition = observed_definition or {}


    def compile(self, options: ALPOptions, catalog: Catalog) -> CompiledProfile:
        operations = tuple(op for op in OPERATION_DEFS if op in options.allowed_operations)
        if not set(operations).issubset(catalog.allowed_operations):
            raise ALPError("OPERATION_NOT_AVAILABLE", "An operation is outside this catalog.")
        if isinstance(options.choice, OperationChoice):
            operations = (options.choice.operation,)
        digest = hashlib.sha256(
            stable_json(
                {
                    "manifest": self.contracts.manifest,
                    "operations": operations,
                    "catalog": catalog.model_dump(),
                    "compiler": self.compiler_identity,
                }
            ).encode()
        ).hexdigest()
        with self._lock:
            if digest in self._cache:
                self._cache.move_to_end(digest)
                return self._cache[digest]
            profile = self._compile(digest, operations, catalog)
            self._cache[digest] = profile
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return profile


    def _compile(self, digest, operations, catalog):
        return CompiledProfile(
            digest, operations, {op: self._specialize(op, catalog) for op in operations},
            "", (), catalog.model_copy(deep=True),
        )

    def _specialize(self, operation: str, catalog: Catalog) -> dict:
        body = self.contracts.schema_for(operation, "body")
        # ALP ObjectSchema is ValueSchema intersected with type=object. Resolve
        # this exact finite union instead of dropping the whole allOf and losing
        # mandatory properties/required/additionalProperties in schema values.
        definitions = body.get("$defs", {})
        if definitions.get("ObjectSchema") == {
            "allOf": [
                {"$ref": "#/$defs/ValueSchema"},
                {"properties": {"type": {"const": "object"}}},
            ]
        }:
            candidates = [
                branch
                for branch in definitions["ValueSchema"]["oneOf"]
                if branch.get("properties", {}).get("type") == {"const": "object"}
            ]
            if len(candidates) == 1:
                definitions["ObjectSchema"] = copy.deepcopy(candidates[0])
        payload_name = OPERATION_DEFS[operation][1]
        base = self.contracts.protocol_schema["$defs"][payload_name]
        variants = []
        if operation == "agent_call":
            for name, agent in catalog.agents.items():
                payload = copy.deepcopy(base)
                if catalog.explicit_session_mode:
                    payload["required"] = [*payload["required"], "session_mode"]
                payload["properties"]["instance_id"] = {"const": name}
                input_schema = copy.deepcopy(self.contracts.protocol_schema["$defs"]["Input"])
                input_schema["properties"]["arguments"] = _import_schema(body, agent.input_schema)
                input_schema["required"] = ["task"]
                # ALP defaults omitted arguments to {}. Require the field only
                # when that default does not satisfy this agent's input schema.
                if not Draft202012Validator(agent.input_schema).is_valid({}):
                    input_schema["required"].append("arguments")
                payload["properties"]["input"] = input_schema
                variants.append(payload)
        elif operation == "agent_capability_call":
            for name, agent in catalog.agents.items():
                for cap_name, capability in agent.capabilities.items():
                    payload = copy.deepcopy(base)
                    payload["properties"]["instance_id"] = {"const": name}
                    payload["properties"]["capability"] = {"const": cap_name}
                    payload["properties"]["arguments"] = _import_schema(
                        body, capability.input_schema
                    )
                    variants.append(payload)
        elif operation == "list_agent_capabilities":
            if catalog.agents:
                payload = copy.deepcopy(base)
                payload["properties"]["instance_id"] = {"enum": list(catalog.agents)}
                variants.append(payload)
        elif operation == "tool_call":
            from .host_contracts import tool_argument_variants

            for name, tool in catalog.tools.items():
                for argument_schema in tool_argument_variants(name, tool, catalog):
                    payload = copy.deepcopy(base)
                    payload["properties"]["tool"] = {"const": name}
                    payload["properties"]["arguments"] = _import_schema(body, argument_schema)
                    variants.append(payload)
        elif operation == "agent_final":
            payload = copy.deepcopy(base)
            value = {"type": "string", "maxLength": 65536}
            if catalog.final_format == "json":
                value = _import_schema(body, catalog.final_schema)
            payload["properties"]["output"] = {
                "type": "object",
                "properties": {
                    "format": {"const": catalog.final_format},
                    "value": value,
                },
                "required": ["format", "value"],
                "additionalProperties": False,
            }
            variants.append(payload)
        else:
            payload = copy.deepcopy(base)
            payload["properties"]["requested_tools"]["items"] = (
                {"enum": list(catalog.tools)} if catalog.tools else False
            )
            variants.append(payload)
        if not variants:
            raise ALPError(
                "EMPTY_OPERATION_DOMAIN", f"The catalog has no targets for {operation}.", 422
            )
        body["properties"]["payload"] = variants[0] if len(variants) == 1 else {"anyOf": variants}
        if operation in catalog.payload_constraints:
            body = constrain_payload(body, catalog.payload_constraints[operation])
        if operation == "agent_definition_generate":
            from .definition_constraints import specialize_definition

            dependencies = catalog
            if self.observed_definition:
                from .catalog import PayloadConstraints

                dependencies = catalog.model_copy(deep=True)
                constraints = dependencies.payload_constraints.get(operation, PayloadConstraints())
                dependencies.payload_constraints[operation] = constraints.model_copy(update={
                    "fixed_values": {**constraints.fixed_values, **{
                        "/" + key: value for key, value in self.observed_definition.items()
                    }},
                })
            body = specialize_definition(body, dependencies, self.contracts)
            # Specialization may replace dependent leaf schemas. Reapply trusted
            # requirements last so no later transform can erase a host constant.
            if operation in catalog.payload_constraints:
                body = constrain_payload(body, catalog.payload_constraints[operation])
        return body
