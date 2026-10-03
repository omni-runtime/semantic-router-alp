#!/usr/bin/env python3
"""Verify the deployed SR native ALP endpoint with an authorized external suite.

Requires the private alp_schema_mcp dependency and its producer test suite.
Reports are local artifacts. This generates calls; it never executes their actions.
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import httpx
from alp_schema_mcp.catalog import ContractCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="cloud-only")
    parser.add_argument("--api-key-env", default="ALP_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--adapter-object-instructions", action="store_true",
                        help="Retain the suite's JSON adapter-object formatting instruction for a legacy comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("alp_checker", args.suite / "examples/check_producer.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    contracts = ContractCatalog()
    suite = checker.load_suite(contracts)
    rows = []
    with httpx.Client(base_url=args.base_url, timeout=600, trust_env=False,
                      headers={"Authorization": "Bearer " + os.environ[args.api_key_env]}) as client:
        for case in suite["positive"]:
            operation = next(x["equals"] for x in case["assertions"] if x["path"] == "/operation")
            request = {"model": args.model, "messages": [{"role": "user", "content":
                suite["producer_instructions"] + ("\n" + suite["format_instructions"]["api"] if args.adapter_object_instructions else "") +
                "\nContext:\n" + case["context"] + "\nTask:\n" + case["prompt"]}],
                "alp": {"allowed_operations": [operation], "choice": {"operation": operation},
                    "catalog_ref": "conformance-json" if case["id"] == "P16" else "conformance-text"},
                "stream": True, "include_raw": True, "temperature": 0.0, "seed": 42,
                "max_tokens": args.max_tokens}
            start = time.monotonic()
            events, error, response, status = [], None, None, 0
            try:
                result = client.post("/v1/alp/chat/completions", json=request)
                status = result.status_code
                if status == 200:
                    for line in result.text.splitlines():
                        if line.startswith("data: ") and line[6:] != "[DONE]":
                            events.append(json.loads(line[6:]))
                    if events and events[-1].get("type") == "agent_call.completed":
                        response = events[-1]["response"]
                else:
                    error = result.text
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                error = type(exc).__name__
            # The original API fixture tests a JSON adapter descriptor, not a
            # provider-native message. Compare the same task assertions against
            # canonical output and independently verify lossless native mapping.
            canonical = response["choices"][0]["message"]["agent_calls"][0]["request"] if response else None
            checked = checker.check_output(json.dumps(canonical, ensure_ascii=False) if canonical else "", "canonical", contracts, case)
            raw_matches = False
            if response:
                try:
                    raw = json.loads(response["raw"])
                    arguments = raw["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    expected = {k: canonical[k] for k in ("protocol_version", "request_id")}
                    if response.get("alp", {}).get("provider_projection") == "typed":
                        expected["payload"] = canonical["payload"]
                    else:
                        expected["payload_json"] = arguments["payload_json"]
                        assert json.loads(arguments["payload_json"]) == canonical["payload"]
                    raw_matches = arguments == expected and raw["name"] == contracts.binding(canonical["operation"])["api_function"]
                except (ValueError, KeyError, TypeError, AssertionError):
                    raw_matches = False
            checked.pop("canonical", None)
            # A rejected upstream call has no canonical document to score. Keep
            # its actual adapter/HTTP error instead of diagnosing an empty JSON
            # string produced by this verifier as the model's protocol failure.
            failure_stage = None
            if response is None:
                failure_stage = "transport" if status == 0 or status in (408, 429, 503, 504) else "adapter"
                try:
                    server_error = json.loads(error or "{}").get("error", {})
                except (ValueError, AttributeError):
                    server_error = {}
                checked["static_valid"] = None
                checked["scenario_checked"] = False
                checked["errors"] = [{
                    "code": server_error.get("code", "HTTP_ERROR" if status else "TRANSPORT_ERROR"),
                    "message": server_error.get("message", "No accepted canonical response was returned."),
                }] if isinstance(server_error, dict) else [{"code": "HTTP_ERROR", "message": "No accepted canonical response was returned."}]
            elif not checked["passed"]:
                failure_stage = "protocol" if not checked["static_valid"] else "task"
            accepted = bool(response and response.get("alp", {}).get("validated") is True
                            and response["alp"].get("authorized") is False
                            and response["alp"].get("executed") is False)
            row = {"id": case["id"], **checked, "accepted": accepted,
                   "passed": checked["passed"] and accepted and raw_matches, "native_mapping_exact": raw_matches, "http_status": status,
                   "seconds": round(time.monotonic() - start, 3), "failure_stage": failure_stage, "transport_error": error}
            rows.append(row)
            (args.output / (case["id"] + ".json")).write_text(json.dumps(
                {"request": request, "events": events, "error": error}, ensure_ascii=False, indent=2))
            report = {"mode": "deployed_SR_cloud_function_calling", "model": args.model,
                      "codec": "provider_native", "task_comparison": "external checker field semantics; task/context unchanged", "adapter_object_instructions": args.adapter_object_instructions,
                      "max_tokens": args.max_tokens, "seed": 42, "retry_count": 0,
                      "passed": sum(x["passed"] for x in rows), "accepted": sum(x["accepted"] for x in rows),
                      "total": len(rows), "expected_total": len(suite["positive"]),
                      "run_complete": len(rows) == len(suite["positive"]),
                      "results": rows, "executed": False, "authorized": False}
            (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return int(not all(x["passed"] for x in rows))


if __name__ == "__main__":
    raise SystemExit(main())
