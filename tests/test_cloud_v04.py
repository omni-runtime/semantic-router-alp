import copy
import json

import pytest
from alp_schema_mcp.catalog import ContractCatalog
from semantic_router_alp.catalog import ServerConfig
from semantic_router_alp.cloud import CloudAdapter
from semantic_router_alp.errors import ALPError

C = ContractCatalog('0.4.0')


def setup(projection):
    adapter = CloudAdapter(ServerConfig(catalogs={'test': {'allowed_operations': ['list_agent_capabilities', 'agent_final'],
        'agents': {'agt_001': {'input_schema': {'type': 'object'}}}}}), projection=projection)
    body = {'model': 'test', 'messages': [{'role': 'user', 'content': 'list twice'}],
            'alp': {'protocol_version': '0.4.0', 'allowed_operations': ['list_agent_capabilities', 'agent_final'], 'catalog_ref': 'test'}, 'include_raw': True}
    prepared = adapter.prepare(json.dumps(body))
    assert prepared['body']['parallel_tool_calls']
    calls = []
    for i in range(2):
        arguments = {'protocol_version': '0.4.0', 'request_id': f'r{i}'}
        payload = {'instance_id': 'agt_001'}
        arguments['payload' if projection == 'typed' else 'payload_json'] = payload if projection == 'typed' else json.dumps(payload)
        calls.append({'id': f'provider_{i}', 'type': 'function', 'function': {'name': 'alp_list_agent_capabilities', 'arguments': json.dumps(arguments)}})
    output = {'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': None, 'tool_calls': calls}}]}
    return adapter, body, prepared, output


@pytest.mark.parametrize('projection', ['typed', 'api_json'])
def test_multiple_native_calls_and_complete_private_pairing(projection):
    adapter, body, prepared, output = setup(projection)
    result = adapter.complete(prepared['context'], json.dumps(output))
    accepted = result['response']['choices'][0]['message']['agent_calls']
    assert len(accepted) == len({call['id'] for call in accepted}) == 2
    assert [p['provider_call_id'] for p in result['history']['pending']] == ['provider_0', 'provider_1']
    assert 'provider_0' not in json.dumps(result['response'])
    success = next(copy.deepcopy(v) for v in C.examples.values() if v.get('operation') == 'list_agent_capabilities' and v.get('ok') is True)
    messages = []
    for i, call in enumerate(accepted):
        value = copy.deepcopy(success)
        value['request_id'] = f'r{i}'
        value['data']['instance_id'] = 'agt_001'
        messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(value)})
    next_body = {**body, 'messages': messages + body['messages']}
    following = adapter.prepare(json.dumps(next_body), history=result['history'])
    assert [m['tool_call_id'] for m in following['body']['messages'] if m['role'] == 'tool'] == ['provider_0', 'provider_1']
    for bad in [messages[:1], messages[::-1], messages + messages[:1]]:
        with pytest.raises(ALPError): adapter.prepare(json.dumps({**body, 'messages': bad}), history=result['history'])
    bad = copy.deepcopy(next_body)
    value = json.loads(bad['messages'][0]['content']); value['data']['instance_id'] = 'wrong'
    bad['messages'][0]['content'] = json.dumps(value)
    with pytest.raises(ALPError): adapter.prepare(json.dumps(bad), history=result['history'])
    bad = copy.deepcopy(next_body); bad['alp']['protocol_version'] = '0.3.0'
    with pytest.raises(ALPError, match='version'): adapter.prepare(json.dumps(bad), history=result['history'])


@pytest.mark.parametrize('projection', ['typed', 'api_json'])
@pytest.mark.parametrize('defect', ['provider_duplicate', 'request_duplicate', 'invalid_second', 'truncated', 'mixed_final'])
def test_invalid_collection_never_returns_prefix_or_history(projection, defect):
    adapter, _body, prepared, output = setup(projection)
    calls = output['choices'][0]['message']['tool_calls']
    args = json.loads(calls[1]['function']['arguments'])
    if defect == 'provider_duplicate': calls[1]['id'] = calls[0]['id']
    if defect == 'request_duplicate': args['request_id'] = 'r0'
    if defect == 'invalid_second': args['extra'] = True
    if defect == 'truncated': output['choices'][0]['finish_reason'] = 'length'
    if defect == 'mixed_final':
        calls[1]['function']['name'] = 'alp_agent_final'
        payload = {'output': {'format': 'text', 'value': 'done'}}
        args['payload' if projection == 'typed' else 'payload_json'] = payload if projection == 'typed' else json.dumps(payload)
    calls[1]['function']['arguments'] = json.dumps(args)
    with pytest.raises(ALPError): adapter.complete(prepared['context'], json.dumps(output))


@pytest.mark.parametrize('projection', ['typed', 'api_json'])
def test_explicit_format_repair_preserves_private_ids_and_is_bounded(projection):
    adapter, body, prepared, output = setup(projection)
    bad = copy.deepcopy(output)
    args = json.loads(bad['choices'][0]['message']['tool_calls'][1]['function']['arguments'])
    args['unexpected'] = True
    bad['choices'][0]['message']['tool_calls'][1]['function']['arguments'] = json.dumps(args)
    with pytest.raises(ALPError) as rejected:
        adapter.complete(prepared['context'], json.dumps(bad))
    history = rejected.value.private_history
    assert history['repair']['remaining'] == 1 and history['pending'] == []
    assert 'provider_0' not in json.dumps(rejected.value.envelope())
    retry = adapter.prepare(json.dumps(body), history=history)
    messages = retry['body']['messages']
    assert [m['tool_call_id'] for m in messages if m['role'] == 'tool'] == ['provider_0', 'provider_1']
    assert sum(m['role'] == 'user' for m in messages) == 1
    assert all(json.loads(m['content'])['executed'] is False for m in messages if m['role'] == 'tool')
    success = adapter.complete(retry['context'], json.dumps(output))
    assert len(success['response']['choices'][0]['message']['agent_calls']) == 2
    with pytest.raises(ALPError) as failed_again:
        adapter.complete(retry['context'], json.dumps(bad))
    assert failed_again.value.private_history['repair']['remaining'] == 0
    with pytest.raises(ALPError, match='already used'):
        adapter.prepare(json.dumps(body), history=failed_again.value.private_history)
    changed = copy.deepcopy(body); changed['messages'][0]['content'] = 'another task'
    with pytest.raises(ALPError, match='original request'):
        adapter.prepare(json.dumps(changed), history=history)
    for defect in ['duplicate', 'truncated']:
        broken = copy.deepcopy(bad)
        if defect == 'duplicate':
            broken['choices'][0]['message']['tool_calls'][1]['id'] = 'provider_0'
        else:
            broken['choices'][0]['finish_reason'] = 'length'
        with pytest.raises(ALPError) as exc:
            adapter.complete(prepared['context'], json.dumps(broken))
        assert not hasattr(exc.value, 'private_history')


@pytest.mark.parametrize('count', [1, 2, 16])
def test_ordered_native_tools_expose_root_fields_and_accept_exact_members(count):
    from test_response_constraints import make
    cat, _, actions = make(count)
    adapter = CloudAdapter(ServerConfig(catalogs={'test': cat}), projection='typed')
    body = {'model':'test', 'messages':[{'role':'user','content':'Call the agents'}],
            'alp':{'protocol_version':'0.4.0','catalog_ref':'test','allowed_operations':['agent_call']}}
    prepared=adapter.prepare(json.dumps(body))
    schema=prepared['body']['tools'][0]['function']['parameters']
    assert schema['type']=='object'
    assert set(schema['properties'])=={'protocol_version','request_id','payload'}
    assert schema['properties']['payload']['type']=='object'
    assert len(json.dumps(schema)) < 50000  # repeated definitions must not consume the model context
    calls=[{'id':f'p{i}','type':'function','function':{'name':'alp_agent_call',
            'arguments':json.dumps({k:v for k,v in a.items() if k!='operation'})}} for i,a in enumerate(actions)]
    output={'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':calls}}]}
    response=adapter.complete(prepared['context'],json.dumps(output))['response']
    assert len(response['choices'][0]['message']['agent_calls'])==count


@pytest.mark.parametrize('projection', ['typed', 'api_json'])
@pytest.mark.parametrize('defect', ['count', 'envelope'])
def test_format_repair_keeps_actionable_count_and_envelope_diagnostics(projection, defect):
    from semantic_router_alp.catalog import ResponseConstraints
    adapter, body, _, output = setup(projection)
    adapter.config.catalogs['test'].response_constraints = ResponseConstraints(min_calls=2, max_calls=2)
    prepared = adapter.prepare(json.dumps(body))
    calls = output['choices'][0]['message']['tool_calls']
    field = 'payload' if projection == 'typed' else 'payload_json'
    if defect == 'count':
        calls.pop()
    else:
        args = json.loads(calls[0]['function']['arguments'])
        args.pop(field)
        calls[0]['function']['arguments'] = json.dumps(args)
    with pytest.raises(ALPError) as rejected:
        adapter.complete(prepared['context'], json.dumps(output))
    tools = [m for m in rejected.value.private_history['messages'] if m['role'] == 'tool']
    feedback = json.loads(tools[0]['content'])
    assert feedback['executed'] is False
    detail = feedback['alp_local_error']['details'][0]
    if defect == 'count':
        assert (detail['expected_min'], detail['expected_max'], detail['actual_count']) == (2, 2, 1)
    elif projection == 'typed':
        assert detail['missing_fields'] == [field]
        assert detail['unexpected_field_count'] == 0
    else:
        assert detail['code'] == 'INVALID_API_ARGUMENTS'


@pytest.mark.parametrize('projection', ['typed', 'api_json'])
@pytest.mark.parametrize('arguments', ['{"unfinished":', '[1,2]', 'null', '{"a":1,"a":2}', '"encoded"'])
def test_bad_json_is_diagnostic_only_and_never_replayed_as_a_native_call(projection, arguments):
    adapter, body, prepared, output = setup(projection)
    bad = copy.deepcopy(output)
    bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = arguments
    with pytest.raises(ALPError) as rejected:
        adapter.complete(prepared['context'], json.dumps(bad))
    history = rejected.value.private_history
    assert history['rejected_provider_response'] == bad
    retry = adapter.prepare(json.dumps(body), history=history)
    assert not any(m.get('tool_calls') or m['role'] == 'tool' for m in retry['body']['messages'])
    assert retry['body']['messages'][1] == body['messages'][0]
    feedback = json.loads(retry['body']['messages'][0]['content'])['format_repair']
    assert feedback['executed'] is False
    assert feedback['expected_argument_fields'][-1] == ('payload' if projection == 'typed' else 'payload_json')
    result = adapter.complete(retry['context'], json.dumps(output))
    assert len(result['response']['choices'][0]['message']['agent_calls']) == 2
    assert 'rejected_provider_response' not in result['response']
    assert result['response']['alp']['provider_finish_reason'] == 'tool_calls'


def test_typed_json_error_includes_only_local_position_and_envelope_diagnostics():
    adapter, _, prepared, output = setup('typed')
    output['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{\n "private_value":'
    with pytest.raises(ALPError) as rejected:
        adapter.complete(prepared['context'], json.dumps(output))
    detail = rejected.value.details[0]
    assert detail['line'] == 2 and detail['column'] > 1 and detail['action_index'] == 0
    assert 'private_value' not in json.dumps(rejected.value.envelope())


def test_tool_choice_is_an_explicit_provider_policy():
    adapter, body, _, _ = setup('typed')
    body['alp']['allowed_operations'] = ['list_agent_capabilities']
    assert isinstance(adapter.prepare(json.dumps(body))['body']['tool_choice'], dict)
    adapter.config.native_tool_choice_policy = 'required'
    assert adapter.prepare(json.dumps(body))['body']['tool_choice'] == 'required'
    adapter.config.native_tool_choice_policy = 'named'
    body['alp']['allowed_operations'].append('agent_final')
    with pytest.raises(ALPError) as bad:
        adapter.prepare(json.dumps(body))
    assert bad.value.code == 'INVALID_PROVIDER_POLICY'


def test_cloud_guidance_does_not_supply_case_values_or_replace_original_task():
    adapter, body, _, _ = setup('typed')
    prepared = adapter.prepare(json.dumps(body))
    context = json.loads(prepared['body']['messages'][0]['content'])
    assert 'parameter_fidelity' in context['generation_guidance']
    assert context['generation_guidance']['parameter_fidelity'].startswith('For tasks requesting exact parameter encoding')
    assert prepared['body']['messages'][1:] == body['messages']
    adapter.config.catalogs['test'].semantic_rendering = False
    disabled = adapter.prepare(json.dumps(body))
    assert json.loads(disabled['body']['messages'][0]['content'])['generation_guidance'] == {}
