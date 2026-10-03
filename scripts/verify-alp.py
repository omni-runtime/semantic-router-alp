#!/usr/bin/env python3
"""Run an authorized producer suite with optional explicit format repair; never execute actions."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import httpx
from alp_schema_mcp.catalog import ContractCatalog


def operations_from_task(case):
    # Published task inputs, never golden files or expected outputs.
    marker = '本轮的业务约束：'
    if marker in case['context']:
        business = json.JSONDecoder().raw_decode(case['context'].split(marker, 1)[1].lstrip())[0]
        return list(dict.fromkeys(item['operation'] for item in business))
    ops = {**dict.fromkeys(range(1, 5), 'agent_definition_generate'), 5: 'list_agent_capabilities',
           **dict.fromkeys([6, 7, 8, 18, 19], 'agent_call'), 9: 'agent_capability_call', 10: 'agent_capability_call',
           **dict.fromkeys([11, 12, 13, 14, 20], 'tool_call'), **dict.fromkeys([15, 16, 17], 'agent_final')}
    number = int(case['id'][1:])
    if number not in ops:
        raise ValueError('Supply --case-config operations for this new task')
    return [ops[number]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--codec', choices=['tagged', 'canonical', 'provider_native'], default='provider_native')
    parser.add_argument('--protocol-version', choices=['0.3.0', '0.4.0'], default='0.3.0')
    parser.add_argument('--api-key-env', default='ALP_API_KEY')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-config', type=Path, help='Operator JSON mapping case IDs to catalog_ref and operations')
    parser.add_argument('--case', action='append')
    parser.add_argument('--max-tokens', type=int, default=3072)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--repair-once', action='store_true', help='Explicitly consume one server-issued format-repair token; retain both attempts')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    version04 = args.protocol_version == '0.4.0'
    checker_path = args.suite / 'examples' / ('check_producer_v04.py' if version04 else 'check_producer.py')
    spec = importlib.util.spec_from_file_location('alp_producer_checker', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    contracts = ContractCatalog(args.protocol_version)
    suite = checker.load_suite(contracts)
    settings = json.loads(args.case_config.read_text()) if args.case_config else {}
    results = []
    with httpx.Client(base_url=args.base_url, timeout=args.timeout, trust_env=False,
                      headers={'Authorization': 'Bearer ' + os.environ[args.api_key_env]}) as client:
        for case in suite['positive']:
            if args.case and case['id'] not in args.case:
                continue
            config = settings.get(case['id'], {})
            operations = config.get('operations') or operations_from_task(case)
            prompt = suite['producer_instructions']
            if args.codec != 'provider_native':
                prompt += '\n' + suite['format_instructions'][args.codec]
            prompt += '\nContext:\n' + case['context'] + '\nTask:\n' + case['prompt']
            body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
                    'alp': {'protocol_version': args.protocol_version, 'allowed_operations': operations, 'choice': 'required',
                            'catalog_ref': config.get('catalog_ref', 'conformance-json' if int(case['id'][1:]) == 16 else 'conformance-text')},
                    'stream': True, 'include_raw': True, 'max_tokens': args.max_tokens, 'temperature': 0.0, 'seed': 42}
            started = time.monotonic()
            events, response, raw = [], None, ''
            attempts = []
            try:
                upstream = client.post('/v1/alp/chat/completions', json=body)
                status, wire = upstream.status_code, upstream.text
                attempts.append({'http_status': status, 'wire': wire})
                if args.repair_once and status != 200:
                    failure = upstream.json().get('alp', {})
                    token = failure.get('provider_state')
                    if failure.get('format_repair_available') is True and isinstance(token, str):
                        upstream = client.post('/v1/alp/chat/completions', json=body,
                                               headers={'x-alp-provider-state': token})
                        status, wire = upstream.status_code, upstream.text
                        attempts.append({'http_status': status, 'wire': wire})
                if status == 200:
                    if wire.lstrip().startswith('{'):
                        response = json.loads(wire)
                    else:
                        for line in wire.splitlines():
                            if line.startswith('data: ') and line[6:] != '[DONE]':
                                event = json.loads(line[6:]); events.append(event)
                                if event.get('type') == 'agent_call.arguments.delta': raw += event['delta']
                                if event.get('type') == 'agent_call.completed': response = event['response']
                error = None if response else (events[-1] if events else json.loads(wire))
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                status, wire, error = 0, '', {'exception': type(exc).__name__}
            accepted = bool(response and response.get('alp', {}).get('validated') is True)
            scored = None
            if accepted:
                actions = [call['request'] for call in response['choices'][0]['message']['agent_calls']]
                document = json.dumps(actions if version04 else actions[0], ensure_ascii=False)
                scored = checker.check_output(document, 'canonical', contracts, case)
                scored.pop('requests' if version04 else 'canonical', None)
            elif raw and args.codec != 'provider_native':
                scored = checker.check_output(raw, args.codec, contracts, case, **({'termination': 'truncated'} if version04 else {}))
                scored.pop('requests' if version04 else 'canonical', None)
            row = {'id': case['id'], 'http_status': status, 'seconds': round(time.monotonic() - started, 3),
                   'retry_count': max(0, len(attempts) - 1), 'accepted': accepted, 'passed': accepted and bool(scored and scored['passed']), 'score': scored, 'error': error}
            (args.output / (case['id'] + '.json')).write_text(json.dumps({'request': body, 'wire': wire, 'attempts': attempts, 'result': row}, ensure_ascii=False, indent=2))
            results.append(row)
            report = {'protocol_version': args.protocol_version, 'model': args.model, 'total': len(results),
                      'accepted': sum(r['accepted'] for r in results), 'passed': sum(r['passed'] for r in results),
                      'retry_count': sum(r['retry_count'] for r in results), 'executed': False, 'authorized': False, 'results': results}
            (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({'case': case['id'], 'retry_count': max(0, len(attempts) - 1), 'accepted': accepted, 'passed': row['passed'], 'seconds': row['seconds'],
                              'completed': len(results)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
