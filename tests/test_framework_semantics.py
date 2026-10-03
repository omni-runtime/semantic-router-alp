import copy
import json

import pytest
from jsonschema import Draft202012Validator

from semantic_router_alp.catalog import Catalog
from semantic_router_alp.constraints import ContractCompiler
from semantic_router_alp.errors import ALPError
from semantic_router_alp.protocol import ALPChatRequest, ALPOptions
from semantic_router_alp.rendering import render_messages
from semantic_router_alp.schema_tools import generation_views, simplify_native_schema

EMPTY = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def compile_catalog(catalog, operation):
    return ContractCompiler().compile(ALPOptions(protocol_version="0.4.0", catalog_ref="test",
                                                allowed_operations=[operation]), catalog)


@pytest.mark.parametrize("operation", ["tool_call", "agent_capability_call"])
def test_write_requires_observed_context_and_exact_precondition(operation):
    if operation == "tool_call":
        cat = Catalog(allowed_operations=[operation], tools={"state.update": {"input_schema": EMPTY, "state_effect": "write"}})
        payload = {"tool": "state.update", "arguments": {}}
    else:
        cat = Catalog(allowed_operations=[operation], agents={"agt_1": {"input_schema": EMPTY, "capabilities": {
            "update": {"input_schema": EMPTY, "state_effect": "write"}}}})
        payload = {"instance_id": "agt_1", "capability": "update", "arguments": {}}
    with pytest.raises(ALPError) as missing:
        compile_catalog(cat, operation)
    assert missing.value.code == "MISSING_RUNTIME_CONTEXT"
    if operation == "tool_call":
        cat.current_state_version = 7
    else:
        cat.agents["agt_1"].state_version = 7
    profile = compile_catalog(cat, operation)
    validator = Draft202012Validator(profile.body_schemas[operation])
    body = {"protocol_version": "0.4.0", "request_id": "write_1", "payload": payload}
    assert not validator.is_valid(body)
    payload["expected_state_version"] = 1
    assert not validator.is_valid(body)
    payload["expected_state_version"] = 7
    assert validator.is_valid(body)


def test_read_preserves_optional_explicit_version_and_does_not_guess_one():
    cat = Catalog(allowed_operations=["tool_call"], tools={"state.read": {"input_schema": EMPTY, "state_effect": "read"}})
    profile = compile_catalog(cat, "tool_call")
    validator = Draft202012Validator(profile.body_schemas["tool_call"])
    body = {"protocol_version": "0.4.0", "request_id": "read_1", "payload": {"tool": "state.read", "arguments": {}}}
    assert validator.is_valid(body)
    body["payload"]["expected_state_version"] = 9
    assert validator.is_valid(body)
    assert cat.current_state_version is None


@pytest.mark.parametrize("codec", ["tagged", "canonical"])
def test_render_uses_actual_generation_envelope_and_keeps_task(codec):
    cat = Catalog(allowed_operations=["agent_final"], local_render_profile="generation")
    profile = compile_catalog(cat, "agent_final")
    request = ALPChatRequest(model="test", messages=[{"role": "user", "content": "Keep this task unchanged。"}],
        alp={"protocol_version": "0.4.0", "catalog_ref": "test", "allowed_operations": ["agent_final"]})
    rendered = render_messages(request, profile, profile.contracts, codec=codec)
    view = json.loads(rendered[0]["content"])["allowed_action_contracts"][0]["body_schema"]
    assert view == generation_views(profile.body_schemas["agent_final"], "agent_final", codec=codec)[0]
    body = {"protocol_version": "0.4.0", "request_id": "final_1", "payload": {"output": {"format": "text", "value": "done"}}}
    if codec == "canonical":
        assert not Draft202012Validator(view).is_valid(body)
        body["operation"] = "agent_final"
    assert Draft202012Validator(view).is_valid(body)
    assert rendered[1]["content"] == request.messages[0].content
    assert json.loads(rendered[0]["content"])["generation_guidance"]
    cat.semantic_rendering = False
    disabled = compile_catalog(cat, "agent_final")
    assert json.loads(render_messages(request, disabled, disabled.contracts, codec=codec)[0]["content"])["generation_guidance"] == {}


def test_stable_native_profile_does_not_receive_cloud_semantic_prose():
    cat = Catalog(allowed_operations=["agent_final"])
    profile = compile_catalog(cat, "agent_final")
    request = ALPChatRequest(model="test", messages=[{"role": "user", "content": "finish"}],
        alp={"protocol_version": "0.4.0", "catalog_ref": "test", "allowed_operations": ["agent_final"]})
    context = json.loads(render_messages(request, profile, profile.contracts, codec="canonical")[0]["content"])
    assert "generation_guidance" not in context
    assert context["allowed_action_contracts"][0]["body_schema"] == profile.body_schemas["agent_final"]


def test_native_compaction_preserves_references_siblings_and_business_literals():
    original = {"type": "object", "properties": {
        "value": {"$ref": "#/$defs/Small"},
        "bounded": {"$ref": "#/$defs/Small", "maxLength": 3},
        "literal": {"const": {"$ref": "#/$defs/Small"}}},
        "required": ["value"], "additionalProperties": False,
        "$defs": {"Small": {"type": "string", "minLength": 1}, "Unused": {"type": "integer"}}}
    before = copy.deepcopy(original)
    compact = simplify_native_schema(original)
    assert original == before
    assert compact["properties"]["literal"] == original["properties"]["literal"]
    assert compact["properties"]["bounded"] == original["properties"]["bounded"]
    assert "Unused" not in compact["$defs"]
    for candidate in [{}, {"value": ""}, {"value": "a"}, {"value": 1}, {"value": "a", "extra": 0},
                      {"value": "ok", "bounded": "four"}, {"value": "ok", "bounded": "two"},
                      {"value": "ok", "literal": {"$ref": "#/$defs/Small"}}]:
        assert Draft202012Validator(original).is_valid(candidate) == Draft202012Validator(compact).is_valid(candidate)
