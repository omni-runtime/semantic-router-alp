"""ALP 0.4 admission: whole turns, version isolation and bounded collections."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from alp_schema_mcp.catalog import ContractCatalog
from semantic_router_alp.catalog import Catalog
from semantic_router_alp.constraints import ContractCompiler
from semantic_router_alp.errors import ALPError
from semantic_router_alp.parser import ALPParser
from semantic_router_alp.protocol import ALPChatRequest, ALPOptions
from semantic_router_alp.rendering import render_messages

C = ContractCatalog('0.4.0')
EMPTY = {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}


def doc(index=0, operation='list_agent_capabilities'):
    payload = {'instance_id': 'agt_001'}
    if operation == 'agent_final':
        payload = {'output': {'format': 'text', 'value': 'literal </agent_final> 🙂'}}
    if operation == 'tool_call':
        payload = {'tool': 'resource.bindings.update', 'arguments': {'expected_resource_revision': 1,
                   'changes': [{'action': 'remove', 'slot': 'optional_docs'}]}}
    return {'protocol_version': '0.4.0', 'request_id': f'req_{index}', 'operation': operation, 'payload': payload}


def catalog():
    reserved = C.resource_tools['resource.bindings.update']
    return Catalog(allowed_operations=['list_agent_capabilities', 'agent_final', 'tool_call'],
                   agents={'agt_001': {'input_schema': EMPTY}}, tools={'resource.bindings.update': {
                       key: reserved[key] for key in ['input_schema', 'state_effect', 'external_effect']}})


@pytest.fixture(scope='module')
def profile():
    cat = catalog()
    return ContractCompiler().compile(ALPOptions(protocol_version='0.4.0', catalog_ref='test', allowed_operations=cat.allowed_operations), cat)


def encode(actions, codec):
    return json.dumps(actions, ensure_ascii=False, separators=(',', ':')) if codec == 'canonical' else '\n'.join('<' + C.binding(a['operation'])['tag'] + '>' + json.dumps({k:v for k,v in a.items() if k != 'operation'}, ensure_ascii=False, separators=(',', ':')) + '</' + C.binding(a['operation'])['tag'] + '>' for a in actions)


@pytest.mark.parametrize('codec', ['canonical', 'tagged'])
@pytest.mark.parametrize('count', [1, 2, 16])
def test_collects_only_at_normal_completion(profile, codec, count):
    actions = [doc(i) for i in range(count)]
    parser = ALPParser(profile, profile.contracts, codec)
    for char in encode(actions, codec):
        assert parser.feed(char) is None
    assert parser.finish('stop') == actions
    with pytest.raises(ALPError):
        parser.finish('stop')


@pytest.mark.parametrize('codec', ['canonical', 'tagged'])
@pytest.mark.parametrize('defect', ['empty', '17', 'duplicate', 'bad_member', 'final_mixed', 'update_mixed', 'tail', 'truncated', 'length', 'version'])
def test_atomic_rejection(profile, codec, defect):
    actions = [doc(0), doc(1)]
    reason = 'stop'
    if defect == 'empty': actions = []
    if defect == '17': actions = [doc(i) for i in range(17)]
    if defect == 'duplicate': actions[1]['request_id'] = actions[0]['request_id']
    if defect == 'bad_member': actions[1]['payload']['instance_id'] = 'unavailable'
    if defect == 'final_mixed': actions[1] = doc(1, 'agent_final')
    if defect == 'update_mixed': actions[1] = doc(1, 'tool_call')
    if defect == 'version': actions[1]['protocol_version'] = '0.3.0'
    if defect == 'length': reason = 'length'
    raw = encode(actions, codec)
    if defect == 'tail': raw += ' explanation'
    if defect == 'truncated': raw = raw[:-1]
    parser = ALPParser(profile, profile.contracts, codec)
    parser.feed(raw)
    with pytest.raises(ALPError): parser.finish(reason)


@pytest.mark.parametrize('operation', ['agent_final', 'tool_call'])
def test_exclusive_singleton(profile, operation):
    parser = ALPParser(profile, profile.contracts, 'canonical')
    parser.feed(encode([doc(0, operation)], 'canonical'))
    assert parser.finish('stop') == [doc(0, operation)]


def test_compiler_version_is_request_local():
    compiler = ContractCompiler()
    cat = Catalog(allowed_operations=['list_agent_capabilities'], agents={'agt_001': {'input_schema': EMPTY}})
    def build(version):
        p = compiler.compile(ALPOptions(protocol_version=version, catalog_ref='test', allowed_operations=cat.allowed_operations), cat)
        assert p.protocol_version == p.contracts.version == version
        return p.digest
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = list(pool.map(build, ['0.3.0', '0.4.0'] * 3))
    assert len(set(digests)) == 2
    assert compiler.contracts.version == '0.3.0'


def test_reserved_management_contract_cannot_be_replaced():
    cat = catalog()
    cat.tools['resource.bindings.update'].input_schema = EMPTY
    with pytest.raises(ALPError, match='descriptor'):
        ContractCompiler().compile(ALPOptions(protocol_version='0.4.0', catalog_ref='test', allowed_operations=cat.allowed_operations), cat)


def test_history_rejects_missing_result_and_version_downgrade(profile):
    body = {'model': 'test', 'alp': {'protocol_version': '0.4.0', 'catalog_ref': 'test', 'allowed_operations': ['list_agent_capabilities']},
            'messages': [{'role': 'assistant', 'content': None, 'agent_calls': [{'id': str(i), 'request': doc(i)} for i in range(2)]}]}
    request = ALPChatRequest.model_validate(body)
    with pytest.raises(ALPError): render_messages(request, profile, profile.contracts)
    body['alp']['protocol_version'] = '0.3.0'
    with pytest.raises(ValueError): ALPChatRequest.model_validate(body)


def test_whole_response_size_limit(profile):
    parser = ALPParser(profile, profile.contracts)
    with pytest.raises(ALPError, match='128 KiB'):
        parser.feed(' ' * (128 * 1024 + 1))


@pytest.mark.parametrize('bound', [False, True])
def test_unavailable_knowledge_tools_cannot_create_an_unfinishable_prefix(bound):
    from jsonschema import Draft202012Validator
    cat = Catalog(allowed_operations=['agent_definition_generate'],
                  resource_bindings=[{'kind': 'knowledge', 'slot': 'docs', 'profile_ref': 'retrieval.docs'}] if bound else None)
    profile = ContractCompiler().compile(ALPOptions(protocol_version='0.4.0', catalog_ref='test',
                                 allowed_operations=cat.allowed_operations), cat)
    body = {'protocol_version': '0.4.0', 'request_id': 'r', 'payload': {
        'resource_requirements': [{'kind': 'knowledge', 'slot': 'docs', 'description': 'Docs',
                                  'required': True, 'profile_ref': 'retrieval.docs', 'access_mode': 'retrieval'}],
        'requested_tools': [], 'name': 'helper', 'description': 'A helper', 'instructions': 'Summarize.'}}
    assert not Draft202012Validator(profile.body_schemas['agent_definition_generate']).is_valid(body)
    body['payload']['resource_requirements'] = []
    assert Draft202012Validator(profile.body_schemas['agent_definition_generate']).is_valid(body)
