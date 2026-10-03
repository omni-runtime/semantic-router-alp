"""Provider-native ALP projection. No network calls or action execution here."""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from types import SimpleNamespace

from alp_schema_mcp.validation import MAX_REQUEST_BYTES, validate_document
from jsonschema import Draft202012Validator
from referencing import Registry
from starlette.datastructures import Headers

from semantic_router_alp.catalog import Catalog, ServerConfig, stable_json
from semantic_router_alp.constraints import ContractCompiler
from semantic_router_alp.errors import ALPError
from semantic_router_alp.host_contracts import (
    task_constraint_coverage,
    validation_scope,
)
from semantic_router_alp.parser import ALPParser
from semantic_router_alp.protocol import ALPChatRequest
from semantic_router_alp.rendering import catalog_context
from semantic_router_alp.schema_tools import compact_schema
from semantic_router_alp.task_context import TASK_HEADER, bind_task_constraints

from .cloud_projection import cloud_body_schema, describe_cloud_fields, typed_arguments_schema
from .response_constraints import member_catalog, response_coverage, validate_response
from .semantics import generation_guidance, OPERATION_DESCRIPTIONS

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
        body = cloud_body_schema(profile, operation)
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
        contracts = profile.contracts
        repair = (history or {}).get("repair")
        if repair:
            if repair["remaining"] < 1:
                raise ALPError("REPAIR_LIMIT_EXCEEDED", "The single format-repair attempt was already used.", 409)
            if repair["constraint_digest"] != profile.digest:
                raise ALPError("REPAIR_CONTRACT_CHANGED", "A format repair cannot change its host contract.", 409)
            if repair["request_digest"] != hashlib.sha256(stable_json(request.model_dump()).encode()).hexdigest():
                raise ALPError("REPAIR_REQUEST_CHANGED", "A format repair must repeat the original request.", 409)
        functions = json.loads(contracts.documents["api-strict-tools.example.json"])
        context = {
            "protocol_version": request.alp.protocol_version, "model_output_codec": "provider_native_typed" if self.projection == "typed" else "provider_native",
            "visible_catalog": catalog_context(profile),
            "generation_guidance": generation_guidance(profile),
        }
        if catalog.semantic_rendering:
            context["generation_guidance"]["parameter_fidelity"] = (
                "For tasks requesting exact parameter encoding or verbatim copying: "
                "copy explicitly supplied agent names, session modes, task text, artifacts and revisions exactly. "
                "Include every supplied artifact and every independent requested call. Dependency arrays "
                "may be omitted or empty when the task supplies no dependencies; do not invent resource requirements from tool names. "
                "Do not replace a supplied task with a summary."
            )
            if request.alp.protocol_version == "0.4.0" and "agent_definition_generate" in profile.operations:
                context["generation_guidance"]["definition_collection"] = (
                    "A task may request several new Agent definitions in one response: emit a separate "
                    "call of the same definition function for each requested definition. Runtime registers "
                    "them serially and returns their real IDs later; serial registration does not limit "
                    "this response to one definition. Calling those new instances waits for the next turn."
                )
        if repair and history.get("repair_feedback"):
            context["format_repair"] = history["repair_feedback"]
        if request.alp.protocol_version == "0.4.0":
            plan = catalog.response_constraints
            context["response_collection"] = {"minItems": plan.min_calls if plan else 1, "maxItems": plan.max_calls if plan else 16,
                "unique_request_ids": True, "exclusive": ["agent_final", "resource.bindings.update"]}
        bound_task = bool(catalog.payload_constraints) and catalog.compact_task_context
        if not bound_task and self.projection == "api_json":
            context["allowed_action_contracts"] = [{
                "operation": op, "api_function": contracts.binding(op)["api_function"],
                "body_schema": (describe_cloud_fields(cloud_body_schema(profile, op), op)
                                if catalog.semantic_rendering else cloud_body_schema(profile, op)),
            } for op in profile.operations]
        # Tool-call argument schemas already live in native parameters. Definitions
        # additionally need tool interfaces to describe their exported capabilities.
        if "agent_definition_generate" in profile.operations:
            context["visible_catalog"]["tool_contracts"] = {
                name: tool.model_dump(exclude_none=True) for name, tool in catalog.tools.items()
            }
        messages = [{"role": "system", "content": stable_json(context)}]
        messages.extend(copy.deepcopy(history["messages"]) if repair else self._history_messages(request, history))
        names = {contracts.binding(op)["api_function"] for op in profile.operations}
        tools = [{"type": "function", "function": {k: v for k, v in f.items() if k != "type"}}
                 for f in functions if f["name"] in names]
        operations = {contracts.binding(op)["api_function"]: op for op in profile.operations}
        for tool in tools:
            function = tool["function"]
            if self.projection == "typed":
                operation = operations[function["name"]]
                function["parameters"] = typed_arguments_schema(profile, operation)
                function["description"] = (f"ALP {operation}. " + OPERATION_DESCRIPTIONS[operation]
                    + " Arguments contain exactly protocol_version, request_id, payload; the function name determines the operation.") if catalog.semantic_rendering else f"ALP {operation}. Arguments use the typed ALP contract."
                # Full schemas exceed some providers' strict subsets. Request
                # that mode only by operator opt-in, without claiming enforcement.
                function["strict"] = self.request_strict
            elif bound_task:
                function["parameters"]["properties"]["payload_json"]["description"] = stable_json({
                    "contentMediaType": "application/json",
                    "contentSchema": self._payload_schema(profile, operations[function["name"]]),
                })
        plan = catalog.response_constraints
        choice = {"type": "function", "function": {"name": next(iter(names))}} if len(names) == 1 and not (plan and plan.min_calls > 1) else "required"
        if self.config.native_tool_choice_policy == "required":
            choice = "required"
        elif self.config.native_tool_choice_policy == "named":
            if len(names) != 1:
                raise ALPError("INVALID_PROVIDER_POLICY", "Named tool choice requires exactly one allowed function.", 422)
            choice = {"type": "function", "function": {"name": next(iter(names))}}
        provider = {
            "model": request.model, "messages": messages, "tools": tools,
            "tool_choice": choice, "parallel_tool_calls": request.alp.protocol_version == "0.4.0" and not (plan and plan.max_calls == 1), "stream": False,
            "temperature": request.temperature, "top_p": request.top_p,
            "max_tokens": request.max_tokens, "n": 1,
        }
        if request.seed is not None:
            provider["seed"] = request.seed
        state = {"repair_attempt": bool(repair), "projection": self.projection, "request_strict": self.request_strict,
                 "provider_tool_choice": choice,
                 "request": request.model_dump(), "catalog": catalog.model_dump(),
                 "history": messages[1:], "response_id": "alpchat_" + uuid.uuid4().hex,
                 "call_id": "ac_" + uuid.uuid4().hex}
        return {"body": provider, "context": state}

    def _history_messages(self, request, history):
        from alp_schema_mcp.validation import validate_action_exchange

        from .constraints import contracts_for
        history = copy.deepcopy(history or {})
        version = request.alp.protocol_version
        contracts = contracts_for(version)
        if history and history.get("protocol_version", "0.3.0") != version:
            raise ALPError("PROTOCOL_CHANGED", "Start a new session after changing the ALP version.", 409)
        if history.get("terminal"):
            raise ALPError("TERMINAL_SESSION", "This ALP session has already finalized.", 409)
        messages = history.get("messages", [])
        pending = history.get("pending") or []
        pending = pending if isinstance(pending, list) else [pending]
        results, result_messages = [], []
        for message in request.messages:
            if message.agent_calls:
                raise ALPError("PROVIDER_STATE_REQUIRED", "Replay native calls through provider state.")
            if len(results) < len(pending):
                expected = pending[len(results)]
                if message.role != "tool" or message.tool_call_id != expected["call_id"] or not isinstance(message.content, str):
                    raise ALPError("UNPAIRED_PROVIDER_CALL", "All pending calls require a paired result in original request order.")
                report = validate_document(message.content, "canonical", contracts)
                value = report.get("canonical", {})
                if (not report["valid"] or value.get("request_id") != expected["request_id"]
                        or value.get("operation") != expected["operation"] or "ok" not in value):
                    raise ALPError("INVALID_TOOL_RESULT", "The result does not match the pending ALP call.")
                results.append(value)
                result_messages.append({"role": "tool", "tool_call_id": expected["provider_call_id"], "content": message.content})
                if len(results) == len(pending):
                    if version == "0.4.0":
                        paired = validate_action_exchange([item["request"] for item in pending], results, contracts)
                        if not paired["valid"]:
                            raise ALPError("INVALID_TOOL_RESULT", "The result collection violates ALP pairing.", 400, paired["errors"])
                    messages.extend(result_messages)
            elif message.role == "tool":
                raise ALPError("UNPAIRED_TOOL_RESULT", "No pending provider call matches this result.")
            else:
                messages.append(message.model_dump(exclude_none=True, exclude={"agent_calls"}))
        if len(results) != len(pending):
            raise ALPError("UNPAIRED_PROVIDER_CALL", "Every pending call requires its paired result before the next model turn.")
        if len(messages) > 256 or len(stable_json(messages).encode()) > self.config.max_request_bytes:
            raise ALPError("HISTORY_TOO_LARGE", "The provider history exceeds its configured bound.", 413)
        return messages

    def complete(self, state: dict, body: str, *, allow_stop_tool_call=False):
        try:
            return self._complete(state, body, allow_stop_tool_call=allow_stop_tool_call)
        except ALPError as exc:
            # Private provider replay, never an accepted prefix or a fabricated
            # canonical ALP Result. The caller explicitly requests any repair.
            if exc.code in {"INVALID_ALP_OUTPUT", "INVALID_CALL_ARGUMENTS", "INVALID_HOST_CONTRACT",
                            "HOST_RESPONSE_COUNT_MISMATCH", "HOST_RESPONSE_ORDER_MISMATCH"}:
                history = self._failure_history(state, body, exc)
                if history is not None:
                    exc.private_history = history
            raise

    def _failure_history(self, state, body, error):
        if len(body.encode()) > 2 * MAX_REQUEST_BYTES:
            return None
        try:
            output = strict_loads(body)
            message = output["choices"][0]["message"]
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not 1 <= len(calls) <= 16 or output["choices"][0]["finish_reason"] not in {"tool_calls", "stop"}:
                return None
            ids = [c["id"] for c in calls]
            if any(not isinstance(i, str) or not 1 <= len(i) <= 256 for i in ids) or len(set(ids)) != len(ids):
                return None
            if message.get("role") != "assistant" or message.get("content") not in (None, ""):
                return None
            replayable = True
            for call in calls:
                if call.get("type") != "function" or not isinstance(call.get("function"), dict):
                    return None
                try:
                    arguments = strict_loads(call["function"]["arguments"])
                    replayable &= isinstance(arguments, dict)
                except (ValueError, TypeError, RecursionError):
                    replayable = False
            request = ALPChatRequest.model_validate(state["request"])
            profile = self.compiler.compile(request.alp, Catalog.model_validate(state["catalog"]))
            # Keep actionable local diagnostics, including required cardinality
            # and missing envelope fields. Dropping them leaves the provider
            # with only a generic error code during its single repair attempt.
            details = [{k: d[k] for k in ("code", "path", "action_index", "rule",
                                         "expected_min", "expected_max", "actual_count",
                                         "missing_fields", "unexpected_field_count", "expected_fields",
                                         "line", "column") if k in d}
                       for d in error.details[:32]]
            content = stable_json({"alp_local_error": {"code": error.code, "details": details}, "executed": False,
                "expected_argument_fields": ["protocol_version", "request_id", "payload" if state["projection"] == "typed" else "payload_json"],
                "repair_instruction": "Regenerate the complete response for the original task, including every requested independent call. Correct the reported format; do not execute, append to, or reuse the failed response."})
            if replayable:
                feedback = [copy.deepcopy(message), *[{"role": "tool", "tool_call_id": i, "content": content} for i in ids]]
            else:
                # ALP 0.4 README:331: invalid provider items never enter replay.
                # Keep the original request; feedback contains only local diagnostics.
                feedback = []
            return {"projection": state["projection"], "protocol_version": request.alp.protocol_version,
                    "messages": [*state["history"], *feedback],
                    **({"rejected_provider_response": output, "repair_feedback": strict_loads(content)} if not replayable else {}),
                    "pending": [], "repair": {"remaining": 0 if state.get("repair_attempt") else 1,
                        "constraint_digest": profile.digest,
                        "request_digest": hashlib.sha256(stable_json(request.model_dump()).encode()).hexdigest()}}
        except (ValueError, KeyError, TypeError, IndexError, ALPError):
            return None

    def _complete(self, state: dict, body: str, *, allow_stop_tool_call=False):
        request = ALPChatRequest.model_validate(state["request"])
        profile = self.compiler.compile(request.alp, Catalog.model_validate(state["catalog"]))
        contracts = profile.contracts
        multi = contracts.version == "0.4.0"
        if len(body.encode()) > 2 * MAX_REQUEST_BYTES:
            raise ALPError("OUTPUT_TOO_LARGE", "The provider response exceeds its bound.", 502)
        try:
            output = strict_loads(body)
            choices = output["choices"]
            if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("index") != 0:
                raise ValueError("Expected one choice")
            choice = choices[0]
            message = choice["message"]
            reasons = {"tool_calls", "stop"} if allow_stop_tool_call else {"tool_calls"}
            if not isinstance(message, dict) or choice["finish_reason"] not in reasons or message.get("role") != "assistant":
                raise ValueError("Incomplete provider turn")
            content = message.get("content")
            if message.get("refusal") or not (content is None or isinstance(content, str) and not content.strip()):
                raise ValueError("Refusal or mixed content")
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not 1 <= len(calls) <= (16 if multi else 1):
                raise ValueError("Invalid function collection")
            ids = set()
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("Expected function call")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not 1 <= len(call_id) <= 256 or call_id in ids:
                    raise ValueError("Missing or duplicate provider call ID")
                ids.add(call_id)
                function = call["function"]
                if (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                        or not isinstance(function.get("arguments"), str)):
                    raise TypeError("Expected encoded function arguments")
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, RecursionError):
            raise ALPError("INVALID_PROVIDER_CALL", "Expected a complete native function response.", 502) from None
        projection = state.get("projection", "api_json")
        operations = {contracts.binding(op)["api_function"]: op for op in profile.operations}
        actions, projections = [], []
        plan = profile.catalog.response_constraints
        if plan and not plan.min_calls <= len(calls) <= plan.max_calls:
            validate_response([{} for _ in calls], profile.catalog)
        for index, call in enumerate(calls):
            function = call["function"]
            projections.append({"name": function["name"], "arguments": function["arguments"]})
            if projection == "typed":
                operation = operations.get(function["name"])
                if operation is None:
                    raise ALPError("UNEXPECTED_OPERATION", "The model selected an unavailable operation.", 502)
                if plan and plan.members and operation != plan.members[index].operation:
                    raise ALPError("HOST_RESPONSE_ORDER_MISMATCH", "Native call order violates the host plan.", 502,
                                   [{"action_index": index, "path": "/operation"}])
                try:
                    arguments = strict_loads(function["arguments"])
                except (ValueError, TypeError, RecursionError) as exc:
                    detail = {"action_index": index, "rule": "strict_json_object", "path": "/arguments"}
                    if isinstance(exc, json.JSONDecodeError):
                        detail.update(line=exc.lineno, column=exc.colno)
                    raise ALPError("INVALID_ALP_OUTPUT", "Invalid native function arguments.", 502, [detail]) from None
                if not isinstance(arguments, dict) or set(arguments) != {"protocol_version", "request_id", "payload"}:
                    raise ALPError("INVALID_ALP_OUTPUT", "Invalid native argument fields.", 502,
                                   [{"action_index": index, "rule": "envelope_fields", "path": "/arguments",
                                     "expected_fields": ["protocol_version", "request_id", "payload"],
                                     "missing_fields": sorted({"protocol_version", "request_id", "payload"} - set(arguments)) if isinstance(arguments, dict) else [],
                                     "unexpected_field_count": len(set(arguments) - {"protocol_version", "request_id", "payload"}) if isinstance(arguments, dict) else 0}])
                report = validate_document(stable_json({**arguments, "operation": operation}), "canonical", contracts)
                if not report["valid"]:
                    raise ALPError("INVALID_ALP_OUTPUT", "The native call violates the ALP contract.", 502, report["errors"])
                errors = []
                validator = Draft202012Validator(typed_arguments_schema(profile, operation, index if profile.member_schemas else None), registry=Registry())
                for error in validator.iter_errors(arguments):
                    errors.append({"code": "NATIVE_ARGUMENT_SCHEMA_MISMATCH", "rule": str(error.validator),
                                   "action_index": index, "path": "/" + "/".join(str(x) for x in error.absolute_path)})
                    if len(errors) == 32:
                        break
                if errors:
                    raise ALPError("INVALID_ALP_OUTPUT", "Native arguments violate their declared contract.", 502, errors)
            elif projection == "api_json":
                report = validate_document(stable_json(projections[-1]), "api", contracts)
            else:
                raise ALPError("INVALID_PROJECTION", "Unknown native projection in private state.", 502)
            if not report["valid"]:
                raise ALPError("INVALID_ALP_OUTPUT", "The native call violates the ALP contract.", 502, report["errors"])
            actions.append(report["canonical"])
        # The API projection and canonical response each have an independent
        # whole-response bound. No accepted prefix or private replay is released.
        if len(stable_json(projections if multi else projections[0]).encode()) > MAX_REQUEST_BYTES:
            raise ALPError("OUTPUT_TOO_LARGE", "The ALP response projection exceeds 128 KiB.", 502)
        parser = ALPParser(profile, contracts, codec="canonical")
        parser.feed(stable_json(actions if multi else actions[0]))
        parser.finish("stop")
        history = {"projection": projection, "protocol_version": contracts.version,
                   "messages": [*state["history"], copy.deepcopy(message)]}
        mappings, accepted = [], []
        for index, (call, canonical) in enumerate(zip(calls, actions, strict=True)):
            call_id = state["call_id"] if index == 0 else "ac_" + uuid.uuid4().hex
            mappings.append({"provider_call_id": call["id"], "call_id": call_id,
                             "request_id": canonical["request_id"], "operation": canonical["operation"], "request": canonical})
            accepted.append({"id": call_id, "type": "alp", "request": canonical})
        if actions[0]["operation"] == "agent_final":
            history["messages"].append({"role": "tool", "tool_call_id": calls[0]["id"],
                                        "content": '{"alp_terminal_ack":true}'})
            history["terminal"] = True
        else:
            history["pending"] = mappings if multi else mappings[0]
        response = {
            "id": state["response_id"], "object": "alp.chat.completion", "created": int(time.time()),
            "model": request.model, "choices": [{"index": 0, "finish_reason": "agent_calls",
                "message": {"role": "assistant", "content": None, "agent_calls": accepted}}],
            "alp": {"protocol_version": contracts.version, "transport": "provider_native",
                "constraint_digest": profile.digest, "validated": True, "executed": False,
                "authorized": False, "provider_projection": projection,
                "provider_finish_reason": choice["finish_reason"],
                "provider_tool_choice": state.get("provider_tool_choice"),
                "semantic_rendering": profile.catalog.semantic_rendering,
                "native_schema_view": ("bound_minimal" if all(a["operation"] in profile.catalog.payload_constraints for a in actions)
                                       else "expanded") if projection == "typed" else "api_envelope",
                "transport_whitespace_ignored": content not in (None, ""),
                "generation_constraints": "native_function_schema_requested" if projection == "typed" else "outer_function_schema_requested",
                "provider_strict_enforcement": "not_verified" if state.get("request_strict", projection == "api_json") else "not_requested",
                "raw_codec": "provider_native_typed" if projection == "typed" else "api",
                "response_constraint_coverage": response_coverage(profile.catalog),
                "validation_scope": [validation_scope(a, member_catalog(profile.catalog, i)) for i, a in enumerate(actions)] if multi else validation_scope(actions[0], profile.catalog),
                "task_constraint_coverage": [task_constraint_coverage(a, member_catalog(profile.catalog, i)) for i, a in enumerate(actions)] if multi else task_constraint_coverage(actions[0], profile.catalog)},
        }
        if output.get("usage") is not None:
            response["usage"] = output["usage"]
        if request.include_raw:
            response["raw"] = stable_json(projections if multi else projections[0])
        return {"response": response, "history": history, "stream": request.stream}
