import copy
import json

import pytest
from jsonschema import Draft202012Validator

from semantic_router_alp.catalog import ServerConfig
from semantic_router_alp.cloud import CloudAdapter
from semantic_router_alp.cloud_projection import native_scalar_enums
from semantic_router_alp.errors import ALPError


def prepare(version="0.4.0", projection="typed", **catalog):
    config = ServerConfig(catalogs={"test": {"allowed_operations": ["agent_definition_generate"], **catalog}})
    adapter = CloudAdapter(config, projection=projection)
    request = {"model": "test", "messages": [{"role": "user", "content": "Define a helper."}],
               "alp": {"protocol_version": version, "allowed_operations": ["agent_definition_generate"], "catalog_ref": "test"}}
    prepared = adapter.prepare(json.dumps(request))
    value = {"protocol_version": version, "request_id": "definition_1",
             "payload": {"name": "helper", "description": "Help", "instructions": "Answer questions"}}
    return adapter, prepared, value


def response(value, projection):
    arguments = copy.deepcopy(value)
    if projection == "api_json":
        arguments["payload_json"] = json.dumps(arguments.pop("payload"))
    return json.dumps({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "provider_1", "type": "function", "function": {
            "name": "alp_agent_definition_generate", "arguments": json.dumps(arguments)}}]}}]})


@pytest.mark.parametrize("version", ["0.3.0", "0.4.0"])
@pytest.mark.parametrize("projection", ["typed", "api_json"])
def test_protocol_optional_dependencies_can_be_omitted_without_output_repair(version, projection):
    adapter, prepared, value = prepare(version, projection)
    schema = (prepared["body"]["tools"][0]["function"]["parameters"] if projection == "typed" else
              json.loads(prepared["body"]["messages"][0]["content"])["allowed_action_contracts"][0]["body_schema"])
    assert Draft202012Validator(schema).is_valid(value)
    result = adapter.complete(prepared["context"], response(value, projection))
    action = result["response"]["choices"][0]["message"]["agent_calls"][0]["request"]
    assert action["payload"] == value["payload"]  # No synthesized dependencies/defaults.


def test_cloud_does_not_remove_trusted_required_or_fixed_dependencies():
    adapter, prepared, value = prepare(payload_constraints={"agent_definition_generate": {
        "fixed_values": {"/resource_requirements": []}}})
    with pytest.raises(ALPError):
        adapter.complete(prepared["context"], response(value, "typed"))
    value["payload"]["resource_requirements"] = []
    assert adapter.complete(prepared["context"], response(value, "typed"))["response"]["alp"]["validated"]
    value["payload"]["resource_requirements"] = [{"kind": "mcp"}]
    with pytest.raises(ALPError):
        adapter.complete(prepared["context"], response(value, "typed"))


def test_operator_output_and_prose_limits_survive_cloud_projection():
    adapter, prepared, value = prepare(explicit_definition_output=True, generation_text_limit=4)
    schema = prepared["body"]["tools"][0]["function"]["parameters"]
    validator = Draft202012Validator(schema)
    assert not validator.is_valid(value)
    value["payload"]["output"] = {"format": "text"}
    assert validator.is_valid(value)
    value["payload"]["description"] = "Too long"
    assert not validator.is_valid(value)


def test_field_annotations_are_non_asserting_and_do_not_rewrite_business_schema():
    literal = {"type": "object", "properties": {"name": {"type": "string", "description": "Keep this annotation"}},
               "required": ["name"], "additionalProperties": False}
    _, prepared, value = prepare(payload_constraints={"agent_definition_generate": {
        "fixed_values": {"/input_schema": literal}}})
    schema = prepared["body"]["tools"][0]["function"]["parameters"]
    assert schema["properties"]["payload"]["properties"]["input_schema"]["const"] == literal
    _, enabled, value = prepare()
    _, disabled, _ = prepare(semantic_rendering=False)
    a, b = [p["body"]["tools"][0]["function"]["parameters"] for p in [enabled, disabled]]
    assert "description" in a["properties"]["payload"]["properties"]["name"]
    for sample in [value, {}, {**value, "extra": True}, {**value, "payload": {**value["payload"], "requested_tools": []}}]:
        assert Draft202012Validator(a).is_valid(sample) == Draft202012Validator(b).is_valid(sample)


def test_provider_singleton_enums_keep_exact_values_and_business_literals():
    schema = {"type": "object", "properties": {
        "version": {"type": "string", "const": "0.4.0"},
        "state": {"const": 7}, "enabled": {"const": False},
        "overlap": {"const": "keep", "enum": ["keep", "other"]},
        "business": {"const": {"const": "literal", "properties": {"x": {"const": 1}}}},
    }, "required": ["version"], "additionalProperties": False}
    projected = native_scalar_enums(copy.deepcopy(schema))
    assert projected["properties"]["version"] == {"type": "string", "enum": ["0.4.0"]}
    assert projected["properties"]["business"] == schema["properties"]["business"]
    assert projected["properties"]["overlap"] == schema["properties"]["overlap"]
    values = [{"version": "0.4.0"}, {"version": '"0.4.0"'}, {"version": "0.3.0"},
              {"version": "0.4.0", "state": 7}, {"version": "0.4.0", "state": 8},
              {"version": "0.4.0", "enabled": 0}, {"version": "0.4.0", "enabled": False},
              {"version": "0.4.0", "business": schema["properties"]["business"]["const"]}]
    for value in values:
        assert Draft202012Validator(schema).is_valid(value) == Draft202012Validator(projected).is_valid(value)
