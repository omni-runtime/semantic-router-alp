"""Provider-native ALP projection. No network calls or action execution here."""
from __future__ import annotations

import copy
import json
import time
import uuid
from types import SimpleNamespace

from alp_schema_mcp.validation import MAX_REQUEST_BYTES, validate_document
from jsonschema import Draft202012Validator
from referencing import Registry
from starlette.datastructures import Headers

from alp_schema_mcp.runtime.catalog import Catalog, ServerConfig, stable_json
from .cloud_projection import typed_arguments_schema
from alp_schema_mcp.runtime.constraints import ContractCompiler
from alp_schema_mcp.runtime.errors import ALPError
from alp_schema_mcp.runtime.host_contracts import task_constraint_coverage, validation_scope
from alp_schema_mcp.runtime.parser import ALPParser
from alp_schema_mcp.runtime.protocol import ALPChatRequest
from alp_schema_mcp.runtime.rendering import catalog_context
from alp_schema_mcp.runtime.schema_tools import compact_schema, generation_schemas
from alp_schema_mcp.runtime.task_context import TASK_HEADER, bind_task_constraints


# Canonical specialization is shared with local adapters, without engine imports.
CloudCompiler = ContractCompiler

def strict_loads(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


class CloudAdapter:
    """One generation per call; SR owns dispatch, authentication and private replay."""

    def __init__(self, config: ServerConfig, *, projection="api_json", request_strict=False):
        if projection not in {"api_json", "typed"}:
            raise ValueError("Unknown provider-native ALP projection")
        self.config = config
        self.projection = projection
        self.request_strict = bool(request_strict) or projection == "api_json"
        self.compiler = CloudCompiler(cache_size=config.compiler_cache_size)
        self.contracts = self.compiler.contracts

    @staticmethod
    def _payload_schema(profile, operation):
        """Describe the JSON string's content, not the canonical request envelope.

        Function parameters already define the API envelope. Showing a second,
        canonical envelope here invites providers to serialize that entire body
        into payload_json. Keep all reachable payload assertions and definitions.
        """
        body = generation_schemas(
            profile.body_schemas[operation], operation,
            explicit_output=profile.catalog.explicit_definition_output,
            text_limit=profile.catalog.generation_text_limit,
            instruction_limit=profile.catalog.generation_instruction_limit,
        )[0]
        payload = copy.deepcopy(body["properties"]["payload"])
        payload["$defs"] = copy.deepcopy(body.get("$defs", {}))
        return compact_schema(payload, annotations=False)

    def prepare(self, body: str, headers=None, history=None):
        if history and history.get("projection", "api_json") != self.projection:
            raise ALPError("PROJECTION_CHANGED", "Start a new session after changing the native projection.", 409)
        if len(body.encode()) > self.config.max_request_bytes:
            raise ALPError("REQUEST_TOO_LARGE", "The request exceeds its configured bound.", 413)
        request = ALPChatRequest.model_validate(strict_loads(body))
        catalog = self.config.catalogs.get(request.alp.catalog_ref)
        if catalog is None:
            raise ALPError("UNKNOWN_CATALOG", "The requested catalog is unavailable.", 404)
        raw = SimpleNamespace(headers=Headers({
            TASK_HEADER: headers[TASK_HEADER],
        } if headers and TASK_HEADER in headers else {}))
        catalog = bind_task_constraints(request, catalog, self.config, raw)
        profile = self.compiler.compile(request.alp, catalog)
        functions = json.loads(self.contracts.documents["api-strict-tools.example.json"])
        context = {
            "protocol_version": "0.3.0", "model_output_codec": "provider_native_typed" if self.projection == "typed" else "provider_native",
            "visible_catalog": catalog_context(profile),
        }
        bound_task = bool(catalog.payload_constraints) and catalog.compact_task_context
        if not bound_task and self.projection == "api_json":
            context["allowed_action_contracts"] = [{
                "operation": op, "api_function": self.contracts.binding(op)["api_function"],
                "body_schema": compact_schema(generation_schemas(
                    profile.body_schemas[op], op,
                    explicit_output=profile.catalog.explicit_definition_output,
                    text_limit=profile.catalog.generation_text_limit,
                    instruction_limit=profile.catalog.generation_instruction_limit,
                )[0], annotations=False),
            } for op in profile.operations]
        if set(profile.operations) & {"tool_call", "agent_definition_generate"}:
            context["visible_catalog"]["tool_contracts"] = {
                name: tool.model_dump(exclude_none=True) for name, tool in catalog.tools.items()
            }
        messages = [{"role": "system", "content": stable_json(context)}]
        messages.extend(self._history_messages(request, history))
        names = {self.contracts.binding(op)["api_function"] for op in profile.operations}
        tools = [{"type": "function", "function": {k: v for k, v in f.items() if k != "type"}}
                 for f in functions if f["name"] in names]
        operations = {self.contracts.binding(op)["api_function"]: op for op in profile.operations}
        for tool in tools:
            function = tool["function"]
            if self.projection == "typed":
                operation = operations[function["name"]]
                function["parameters"] = typed_arguments_schema(profile, operation)
                function["description"] = f"ALP {operation}. Arguments use the typed ALP contract."
                # Full schemas exceed some providers' strict subsets. Request
                # that mode only by operator opt-in, without claiming enforcement.
                function["strict"] = self.request_strict
            elif bound_task:
                function["parameters"]["properties"]["payload_json"]["description"] = stable_json({
                    "contentMediaType": "application/json",
                    "contentSchema": self._payload_schema(profile, operations[function["name"]]),
                })
        choice = {"type": "function", "function": {"name": next(iter(names))}} if len(names) == 1 else "required"
        provider = {
            "model": request.model, "messages": messages, "tools": tools,
            "tool_choice": choice, "parallel_tool_calls": False, "stream": False,
            "temperature": request.temperature, "top_p": request.top_p,
            "max_tokens": request.max_tokens, "n": 1,
        }
        if request.seed is not None:
            provider["seed"] = request.seed
        state = {"projection": self.projection, "request_strict": self.request_strict,
                 "request": request.model_dump(), "catalog": catalog.model_dump(),
                 "history": messages[1:], "response_id": "alpchat_" + uuid.uuid4().hex,
                 "call_id": "ac_" + uuid.uuid4().hex}
        return {"body": provider, "context": state}

    def _history_messages(self, request, history):
        history = copy.deepcopy(history or {})
        messages = history.get("messages", [])
        pending = history.get("pending")
        # Prior provider items are supplied only by SR's authenticated private store.
        # Canonical-only assistant history cannot reconstruct reasoning/protocol items.
        for message in request.messages:
            if message.agent_calls:
                raise ALPError("PROVIDER_STATE_REQUIRED", "Replay native calls through provider state.")
            if pending:
                if message.role != "tool" or message.tool_call_id != pending["call_id"] or not isinstance(message.content, str):
                    raise ALPError("UNPAIRED_PROVIDER_CALL", "The previous call requires its paired result.")
                result = validate_document(message.content, "canonical", self.contracts)
                value = result.get("canonical", {})
                if (not result["valid"] or value.get("request_id") != pending["request_id"]
                        or value.get("operation") != pending["operation"] or "ok" not in value):
                    raise ALPError("INVALID_TOOL_RESULT", "The result does not match the pending ALP call.")
                messages.append({"role": "tool", "tool_call_id": pending["provider_call_id"],
                                 "content": message.content})
                pending = None
            elif message.role == "tool":
                raise ALPError("UNPAIRED_TOOL_RESULT", "No pending provider call matches this result.")
            else:
                messages.append(message.model_dump(exclude_none=True, exclude={"agent_calls"}))
        if pending:
            raise ALPError("UNPAIRED_PROVIDER_CALL", "The previous call requires its paired result.")
        if len(messages) > 256 or len(stable_json(messages).encode()) > self.config.max_request_bytes:
            raise ALPError("HISTORY_TOO_LARGE", "The provider history exceeds its configured bound.", 413)
        return messages

    def complete(self, state: dict, body: str, *, allow_stop_tool_call=False):
        if len(body.encode()) > 2 * MAX_REQUEST_BYTES:
            raise ALPError("OUTPUT_TOO_LARGE", "The provider response exceeds its bound.", 502)
        try:
            output = strict_loads(body)
            choices = output["choices"]
            if (not isinstance(choices, list) or len(choices) != 1
                    or not isinstance(choices[0], dict) or choices[0].get("index") != 0):
                raise ValueError("Expected one choice")
            choice = choices[0]
            reasons = {"tool_calls", "stop"} if allow_stop_tool_call else {"tool_calls"}
            message = choice["message"]
            if not isinstance(message, dict) or choice["finish_reason"] not in reasons or message.get("role") != "assistant":
                raise ValueError("Incomplete provider turn")
            content = message.get("content")
            if message.get("refusal") or not (content is None or isinstance(content, str) and not content.strip()):
                raise ValueError("Refusal or mixed content")
            calls = message["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict) or calls[0].get("type") != "function":
                raise ValueError("Expected one function call")
            call = calls[0]
            if not isinstance(call.get("id"), str) or not 1 <= len(call["id"]) <= 256:
                raise ValueError("Missing provider call ID")
            function = call["function"]
            if (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                    or not isinstance(function.get("arguments"), str)):
                raise ValueError("Expected encoded function arguments")
        except (ValueError, KeyError, TypeError, IndexError, RecursionError):
            raise ALPError("INVALID_PROVIDER_CALL", "Expected one complete native function call.", 502) from None
        request = ALPChatRequest.model_validate(state["request"])
        profile = self.compiler.compile(request.alp, Catalog.model_validate(state["catalog"]))
        projection = state.get("projection", "api_json")
        if projection == "typed":
            operations = {self.contracts.binding(op)["api_function"]: op for op in profile.operations}
            operation = operations.get(function["name"])
            if operation is None:
                raise ALPError("UNEXPECTED_OPERATION", "The model selected an unavailable operation.", 502)
            try:
                arguments = strict_loads(function["arguments"])
            except (ValueError, TypeError, RecursionError):
                raise ALPError("INVALID_ALP_OUTPUT", "Invalid native function arguments.", 502) from None
            if not isinstance(arguments, dict) or set(arguments) != {"protocol_version", "request_id", "payload"}:
                raise ALPError("INVALID_ALP_OUTPUT", "Invalid native argument fields.", 502)
            # Apply canonical size/depth limits before walking the specialized
            # generation schema. No network schema retrieval is ever permitted.
            report = validate_document(stable_json({**arguments, "operation": operation}), "canonical", self.contracts)
            if not report["valid"]:
                raise ALPError("INVALID_ALP_OUTPUT", "The native call violates the ALP contract.", 502, report["errors"])
            errors = []
            validator = Draft202012Validator(typed_arguments_schema(profile, operation), registry=Registry())
            for error in validator.iter_errors(arguments):
                errors.append({"code": "NATIVE_ARGUMENT_SCHEMA_MISMATCH", "rule": str(error.validator),
                               "path": "/" + "/".join(str(x) for x in error.absolute_path)})
                if len(errors) == 32:
                    break
            if errors:
                raise ALPError("INVALID_ALP_OUTPUT", "Native arguments violate their declared contract.", 502, errors)
        elif projection == "api_json":
            report = validate_document(stable_json({"name": function["name"], "arguments": function["arguments"]}), "api", self.contracts)
        else:
            raise ALPError("INVALID_PROJECTION", "Unknown native projection in private state.", 502)
        if not report["valid"]:
            raise ALPError("INVALID_ALP_OUTPUT", "The native call violates the ALP contract.", 502, report["errors"])
        parser = ALPParser(profile, self.contracts, codec="canonical")
        parser.feed(stable_json(report["canonical"]))
        canonical = parser.finish("stop")
        history = {"projection": projection, "messages": [*state["history"], copy.deepcopy(message)]}
        mapping = {"provider_call_id": call["id"], "call_id": state["call_id"],
                   "request_id": canonical["request_id"], "operation": canonical["operation"]}
        if canonical["operation"] == "agent_final":
            history["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                        "content": '{"alp_terminal_ack":true}'})
        else:
            history["pending"] = mapping
        response = {
            "id": state["response_id"], "object": "alp.chat.completion", "created": int(time.time()),
            "model": request.model, "choices": [{"index": 0, "finish_reason": "agent_calls",
                "message": {"role": "assistant", "content": None,
                    "agent_calls": [{"id": state["call_id"], "type": "alp", "request": canonical}]}}],
            "alp": {"protocol_version": "0.3.0", "transport": "provider_native",
                "constraint_digest": profile.digest, "validated": True, "executed": False,
                "authorized": False, "provider_projection": projection,
                "native_schema_view": ("bound_minimal" if canonical["operation"] in profile.catalog.payload_constraints
                                       else "expanded") if projection == "typed" else "api_envelope",
                "transport_whitespace_ignored": content not in (None, ""),
                "generation_constraints": "native_function_schema_requested" if projection == "typed" else "outer_function_schema_requested",
                "provider_strict_enforcement": "not_verified" if state.get("request_strict", projection == "api_json") else "not_requested",
                "raw_codec": "provider_native_typed" if projection == "typed" else "api",
                "validation_scope": validation_scope(canonical, profile.catalog),
                "task_constraint_coverage": task_constraint_coverage(canonical, profile.catalog)},
        }
        if output.get("usage") is not None:
            response["usage"] = output["usage"]
        if request.include_raw:
            response["raw"] = stable_json({"name": function["name"], "arguments": function["arguments"]})
        return {"response": response, "history": history, "stream": request.stream}
