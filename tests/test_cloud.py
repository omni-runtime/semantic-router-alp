import copy
import json
import subprocess
import sys

import pytest
from conftest import PAYLOADS, canonical
from conftest import body

from semantic_router_alp.cloud import CloudAdapter
from semantic_router_alp.errors import ALPError
from semantic_router_alp.host_tasks import AgentCallTask, host_task_headers
from semantic_router_alp.protocol import ALPChatRequest


def native(adapter, operation="agent_call", payload=None):
    value = canonical(operation, payload)
    args = {k: value[k] for k in ("protocol_version", "request_id")}
    if adapter.projection == "typed":
        args["payload"] = value["payload"]
    else:
        args["payload_json"] = json.dumps(value["payload"], ensure_ascii=False)
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "provider_1", "type": "function",
            "function": {"name": adapter.contracts.binding(operation)["api_function"],
                         "arguments": json.dumps(args)}}]}}]}


@pytest.mark.parametrize("operation", list(PAYLOADS))
def test_native_operations_use_authoritative_projection(config, operation):
    adapter = CloudAdapter(config)
    prepared = adapter.prepare(json.dumps(body(operation)))
    assert prepared["body"]["parallel_tool_calls"] is False
    assert prepared["body"]["stream"] is False
    assert len(prepared["body"]["tools"]) == 1
    assert prepared["body"]["tools"][0]["function"]["strict"] is True
    result = adapter.complete(prepared["context"], json.dumps(native(adapter, operation)))
    response = result["response"]
    assert response["choices"][0]["message"]["agent_calls"][0]["request"] == canonical(operation)
    assert response["alp"]["validated"] and not response["alp"]["executed"]
    assert "provider_1" not in json.dumps(response)
    if operation == "agent_final":
        assert "pending" not in result["history"]
        assert result["history"]["messages"][-1]["tool_call_id"] == "provider_1"
    else:
        assert result["history"]["pending"]["provider_call_id"] == "provider_1"


@pytest.mark.parametrize("defect", ["truncated", "empty", "multiple", "mixed", "refusal", "bad_payload", "unknown", "duplicate", "double_encoded", "wrong_target"])
def test_reject_bad_native_calls(config, defect):
    adapter = CloudAdapter(config)
    state = adapter.prepare(json.dumps(body("agent_call")))["context"]
    output = native(adapter)
    choice = output["choices"][0]
    message = choice["message"]
    function = message["tool_calls"][0]["function"]
    if defect == "truncated":
        choice["finish_reason"] = "length"
    elif defect == "empty":
        message["tool_calls"] = []
    elif defect == "multiple":
        message["tool_calls"] *= 2
    elif defect == "mixed":
        message["content"] = '<agent_call>{"fake":true}</agent_call>'
    elif defect == "refusal":
        message["refusal"] = "denied"
    elif defect == "bad_payload":
        function["arguments"] = '{}'
    elif defect == "unknown":
        function["name"] = "ordinary_tool"
    elif defect == "duplicate":
        function["arguments"] = '{"protocol_version":"0.3.0","protocol_version":"0.3.0","request_id":"r","payload_json":"{}"}'
    elif defect == "double_encoded":
        args = json.loads(function["arguments"])
        args["payload_json"] = json.dumps(args["payload_json"])
        function["arguments"] = json.dumps(args)
    elif defect == "wrong_target":
        output = native(adapter, payload={"instance_id": "not_in_catalog", "input": {"task": "hello"}})
    with pytest.raises(ALPError):
        adapter.complete(state, json.dumps(output), allow_stop_tool_call=True)


def test_stop_compatibility_is_explicit_and_still_validates(config):
    adapter = CloudAdapter(config)
    state = adapter.prepare(json.dumps(body("agent_call")))["context"]
    output = native(adapter)
    output["choices"][0]["finish_reason"] = "stop"
    with pytest.raises(ALPError):
        adapter.complete(state, json.dumps(output))
    assert adapter.complete(state, json.dumps(output), allow_stop_tool_call=True)
    output["choices"][0]["message"]["tool_calls"] = []
    with pytest.raises(ALPError):
        adapter.complete(state, json.dumps(output), allow_stop_tool_call=True)


def test_cloud_context_uses_shared_definition_generation_contract(config):
    adapter = CloudAdapter(config)
    prepared = adapter.prepare(json.dumps(body("agent_definition_generate")))
    context = json.loads(prepared["body"]["messages"][0]["content"])
    schema = context["allowed_action_contracts"][0]["body_schema"]["properties"]["payload"]
    assert {"resource_requirements", "requested_tools"}.issubset(schema["required"])
    assert not {"protocol_version", "request_id", "payload"} & schema["properties"].keys()
    for name, tool in config.catalogs["product-assistant"].tools.items():
        assert context["visible_catalog"]["tool_contracts"][name] == tool.model_dump(exclude_none=True)


def test_cloud_host_binding_rejects_model_text_drift(config, monkeypatch):
    cfg = config.model_copy(deep=True)
    cfg.task_signing_key_env = "ALP_TEST_KEY"
    key = b"a-test-host-task-key-with-32-bytes!"
    monkeypatch.setenv(cfg.task_signing_key_env, key.decode())
    adapter = CloudAdapter(cfg)
    request = ALPChatRequest.model_validate(body("agent_call"))
    task = AgentCallTask(instance_id="product_assistant_001", task="原样。", session_mode="isolated")
    prepared = adapter.prepare(request.model_dump_json(), host_task_headers(request, task, key=key))
    state = prepared["context"]
    assert "allowed_action_contracts" not in json.loads(prepared["body"]["messages"][0]["content"])
    description = prepared["body"]["tools"][0]["function"]["parameters"]["properties"]["payload_json"]["description"]
    assert json.loads(description)["contentSchema"]["properties"]["input"]["properties"]["task"]["const"] == task.task
    good = adapter.complete(state, json.dumps(native(adapter, payload=task.payload())))
    assert good["response"]["alp"]["task_constraint_coverage"]["key_fields"]["/input/task"] == "fixed"
    changed = task.payload()
    changed["input"]["task"] = "原样"
    with pytest.raises(ALPError):
        adapter.complete(state, json.dumps(native(adapter, payload=changed)))


def test_private_history_retains_provider_items_and_pairs_final(config):
    adapter = CloudAdapter(config)
    state = adapter.prepare(json.dumps(body("agent_final")))["context"]
    output = native(adapter, "agent_final")
    output["choices"][0]["message"]["reasoning_content"] = "private provider history"
    result = adapter.complete(state, json.dumps(output))
    assert "private provider history" not in json.dumps(result["response"])
    next_turn = adapter.prepare(json.dumps(body("agent_call")), history=result["history"])
    assert next_turn["body"]["messages"][-3]["reasoning_content"] == "private provider history"
    assert next_turn["body"]["messages"][-2]["role"] == "tool"


def test_pending_history_requires_exact_result_pair(config):
    adapter = CloudAdapter(config)
    state = adapter.prepare(json.dumps(body("agent_call")))["context"]
    result = adapter.complete(state, json.dumps(native(adapter)))
    with pytest.raises(ALPError, match="paired result"):
        adapter.prepare(json.dumps(body("agent_call")), history=result["history"])
    success = next(copy.deepcopy(v) for v in adapter.contracts.examples.values()
                   if v.get("operation") == "agent_call" and v.get("ok") is True)
    success["request_id"] = "req_test"
    request = body("agent_final")
    request["messages"].insert(0, {"role": "tool", "tool_call_id": state["call_id"], "content": json.dumps(success)})
    prepared = adapter.prepare(json.dumps(request), history=result["history"])
    assert prepared["body"]["messages"][-2]["tool_call_id"] == "provider_1"
    request["messages"][0]["tool_call_id"] = "wrong"
    with pytest.raises(ALPError):
        adapter.prepare(json.dumps(request), history=result["history"])


def test_worker_errors_are_sanitized_and_do_not_crash(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    proc = subprocess.run([sys.executable, "-m", "semantic_router_alp.cloud_worker", "--config", str(path)],
                          input=json.dumps({"action": "prepare", "body": '{"secret":1}'}),
                          text=True, capture_output=True, check=True)
    result = json.loads(proc.stdout)
    assert result["status"] == 400 and not result["ok"]
    assert "secret" not in proc.stdout


@pytest.mark.parametrize("invalid", [{"choices": [None]}, {"choices": [{"index": 0, "message": []}]},
                                    {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                                        "role": "assistant", "tool_calls": [None]}}]}])
def test_malformed_provider_shapes_fail_as_protocol_errors(config, invalid):
    adapter = CloudAdapter(config, projection="typed")
    state = adapter.prepare(json.dumps(body("agent_call")))["context"]
    with pytest.raises(ALPError) as exc:
        adapter.complete(state, json.dumps(invalid))
    assert exc.value.status == 502


@pytest.mark.parametrize("operation", list(PAYLOADS))
def test_typed_projection_roundtrip_and_raw_preservation(config, operation):
    adapter = CloudAdapter(config, projection="typed")
    request = body(operation, include_raw=True)
    prepared = adapter.prepare(json.dumps(request))
    function = prepared["body"]["tools"][0]["function"]
    assert function["strict"] is False
    assert set(function["parameters"]["properties"]) == {"protocol_version", "request_id", "payload"}
    assert "allowed_action_contracts" not in json.loads(prepared["body"]["messages"][0]["content"])
    output = native(adapter, operation)
    original = copy.deepcopy(output)
    result = adapter.complete(prepared["context"], json.dumps(output))
    response = result["response"]
    assert response["choices"][0]["message"]["agent_calls"][0]["request"] == canonical(operation)
    assert response["alp"]["raw_codec"] == "provider_native_typed"
    assert response["alp"]["provider_strict_enforcement"] == "not_requested"
    raw = json.loads(response["raw"])
    assert raw == original["choices"][0]["message"]["tool_calls"][0]["function"]
    assert output == original
    assert result["history"]["projection"] == "typed"


@pytest.mark.parametrize("defect", ["string_payload", "outer_extra", "duplicate", "wrong_target", "bad_arguments", "unknown", "legacy"])
def test_typed_projection_rejects_without_coercion(config, defect):
    adapter = CloudAdapter(config, projection="typed")
    state = adapter.prepare(json.dumps(body("agent_call")))["context"]
    output = native(adapter)
    function = output["choices"][0]["message"]["tool_calls"][0]["function"]
    args = json.loads(function["arguments"])
    if defect == "string_payload":
        args["payload"] = json.dumps(args["payload"])
    elif defect == "outer_extra":
        args["operation"] = "agent_call"
    elif defect == "wrong_target":
        args["payload"]["instance_id"] = "unknown"
    elif defect == "bad_arguments":
        args["payload"]["input"]["arguments"] = {"unauthorized": True}
    elif defect == "unknown":
        function["name"] = "alp_agent_final"
    elif defect == "legacy":
        args["payload_json"] = json.dumps(args.pop("payload"))
    function["arguments"] = json.dumps(args)
    if defect == "duplicate":
        function["arguments"] = '{"request_id":"first",' + function["arguments"][1:]
    with pytest.raises(ALPError):
        adapter.complete(state, json.dumps(output))


def test_typed_exact_host_binding_is_checked_after_generation(config, monkeypatch):
    cfg = config.model_copy(deep=True)
    cfg.task_signing_key_env = "ALP_TEST_KEY"
    key = b"a-test-host-task-key-with-32-bytes!"
    monkeypatch.setenv(cfg.task_signing_key_env, key.decode())
    adapter = CloudAdapter(cfg, projection="typed")
    request = ALPChatRequest.model_validate(body("agent_call"))
    task = AgentCallTask(instance_id="product_assistant_001", task="原样。", session_mode="isolated")
    state = adapter.prepare(request.model_dump_json(), host_task_headers(request, task, key=key))["context"]
    assert adapter.complete(state, json.dumps(native(adapter, payload=task.payload())))
    altered = task.payload()
    altered["input"]["task"] = "原样"
    with pytest.raises(ALPError):
        adapter.complete(state, json.dumps(native(adapter, payload=altered)))


def test_type_annotations_preserve_host_schema_literals():
    from semantic_router_alp.cloud_projection import expose_native_schema

    value = {"anyOf": [{"type": "object"}, {"type": "object"}], "properties": {"x": {"const": "x"}}}
    schema = {"const": copy.deepcopy(value), "enum": [copy.deepcopy(value)],
              "anyOf": [{"type": "object"}, {"type": "object"}]}
    expose_native_schema(schema)
    assert schema["type"] == "object"
    assert schema["const"] == value and schema["enum"] == [value]


@pytest.mark.parametrize("projection", ["api_json", "typed"])
def test_whitespace_is_framing_not_mixed_model_content(config, projection):
    adapter = CloudAdapter(config, projection=projection)
    state = adapter.prepare(json.dumps(body("agent_final")))["context"]
    output = native(adapter, "agent_final")
    output["choices"][0]["message"]["content"] = "\n \t"
    result = adapter.complete(state, json.dumps(output))
    assert result["response"]["alp"]["transport_whitespace_ignored"] is True
    assert result["history"]["messages"][-2]["content"] == "\n \t"
    output["choices"][0]["message"]["content"] = "\n explanation"
    with pytest.raises(ALPError):
        adapter.complete(state, json.dumps(output))


def test_projection_change_requires_new_private_session(config):
    adapter = CloudAdapter(config)
    state = adapter.prepare(json.dumps(body("agent_final")))["context"]
    history = adapter.complete(state, json.dumps(native(adapter, "agent_final")))["history"]
    with pytest.raises(ALPError, match="new session"):
        CloudAdapter(config, projection="typed").prepare(json.dumps(body()), history=history)


def test_provider_strict_opt_in_does_not_claim_enforcement(config):
    adapter = CloudAdapter(config, projection="typed", request_strict=True)
    state = adapter.prepare(json.dumps(body("agent_call")))
    assert state["body"]["tools"][0]["function"]["strict"] is True
    response = adapter.complete(state["context"], json.dumps(native(adapter)))["response"]
    assert response["alp"]["provider_strict_enforcement"] == "not_verified"
    malformed = native(adapter)
    malformed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"bad":1}'
    with pytest.raises(ALPError):
        adapter.complete(state["context"], json.dumps(malformed))


def test_native_schema_visibility_preserves_union_validation():
    from jsonschema import Draft202012Validator

    from semantic_router_alp.cloud_projection import expose_native_schema

    schema = {"$defs": {"Word": {"type": "string"}}, "anyOf": [
        {"type": "object", "properties": {"kind": {"const": "text"}, "value": {"$ref": "#/$defs/Word"}},
         "required": ["kind", "value"], "additionalProperties": False},
        {"type": "object", "properties": {"kind": {"const": "count"}, "value": {"type": "integer"}, "extra": True},
         "required": ["kind", "value"], "additionalProperties": False},
    ]}
    expanded = expose_native_schema(copy.deepcopy(schema))
    assert expanded["type"] == "object" and expanded["properties"]["kind"]["enum"] == ["text", "count"]
    assert expanded["anyOf"][0]["properties"]["value"]["type"] == "string"
    assert expanded["properties"]["extra"] is True
    original, visible = Draft202012Validator(schema), Draft202012Validator(expanded)
    for kind in ("text", "count", "unknown", None):
        for value in ("hello", 1, 1.5, None, {}, [], True):
            for additional in ({}, {"extra": 42}, {"unexpected": 42}):
                candidate = {"kind": kind, "value": value, **additional}
                assert original.is_valid(candidate) == visible.is_valid(candidate)
    for value in ({}, {"kind": "text"}, None, 1, "text", [], True):
        assert original.is_valid(value) == visible.is_valid(value)


def test_bound_native_schema_keeps_host_values_and_union_domain():
    from jsonschema import Draft202012Validator

    from semantic_router_alp.cloud_projection import _expose_bound_types, expose_native_schema

    literal = {"anyOf": [{"type": "object"}, {"type": "object"}]}
    schema = {"anyOf": [
        {"type": "object", "properties": {"name": {"const": "bound"}, "schema": {"const": literal}},
         "required": ["name", "schema"], "additionalProperties": False},
        {"type": "object", "properties": {"name": {"const": "second"}},
         "required": ["name"], "additionalProperties": False},
    ]}
    minimal = copy.deepcopy(schema)
    _expose_bound_types(minimal)
    assert minimal["type"] == "object" and "properties" not in minimal
    assert minimal["anyOf"][0]["properties"]["schema"]["const"] == literal
    assert len(json.dumps(minimal)) < len(json.dumps(expose_native_schema(copy.deepcopy(schema))))
    before, after = Draft202012Validator(schema), Draft202012Validator(minimal)
    for value in ({"name": "bound", "schema": literal}, {"name": "second"},
                  {"name": "bound"}, {"name": "second", "extra": True}, None, [], "bound"):
        assert before.is_valid(value) == after.is_valid(value)


def test_native_schema_visibility_keeps_literals_and_unsafe_unions_intact():
    from jsonschema import Draft202012Validator

    from semantic_router_alp.cloud_projection import expose_native_schema

    literal = {"anyOf": [{"type": "object"}, {"type": "object"}], "$ref": "#/$defs/Foo"}
    schema = {"$defs": {"Loop": {"$ref": "#/$defs/Loop"}}, "properties": {
        "literal": {"const": copy.deepcopy(literal), "default": copy.deepcopy(literal), "enum": [copy.deepcopy(literal)]},
        "cycle": {"$ref": "#/$defs/Loop"},
        "remote": {"$ref": "https://example.invalid/schema"},
    }, "anyOf": [True, {"type": "object"}]}
    expanded = expose_native_schema(copy.deepcopy(schema))
    field = expanded["properties"]["literal"]
    assert field["const"] == literal == field["default"] == field["enum"][0]
    assert expanded["properties"]["cycle"] == schema["properties"]["cycle"]
    assert expanded["properties"]["remote"] == schema["properties"]["remote"]
    assert "additionalProperties" not in expanded
    patterned = {"anyOf": [{"type": "object", "patternProperties": {"^x": {"type": "integer"}},
                            "additionalProperties": False}]}
    expanded = expose_native_schema(copy.deepcopy(patterned))
    assert "additionalProperties" not in expanded
    assert Draft202012Validator(expanded).is_valid({"xyz": 1})


@pytest.mark.parametrize("operation", list(PAYLOADS))
def test_native_projection_keeps_canonical_rejections(config, operation):
    from jsonschema import Draft202012Validator

    from semantic_router_alp.cloud_projection import expose_native_schema

    adapter = CloudAdapter(config, projection="typed")
    request = ALPChatRequest.model_validate(body(operation))
    profile = adapter.compiler.compile(request.alp, config.catalogs[request.alp.catalog_ref])
    original = profile.body_schemas[operation]
    expanded = expose_native_schema(copy.deepcopy(original))
    before, after = Draft202012Validator(original), Draft202012Validator(expanded)
    good = canonical(operation)
    good.pop("operation")
    candidates = [good, {}, None, [], {**good, "extra": 1}, {**good, "payload": "{}"},
                  {**good, "payload": {**good["payload"], "unexpected": True}}]
    for key in good["payload"]:
        bad = copy.deepcopy(good)
        del bad["payload"][key]
        candidates.append(bad)
    for value in candidates:
        assert before.is_valid(value) == after.is_valid(value)
