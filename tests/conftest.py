from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from alp_schema_mcp.runtime.catalog import ServerConfig
from alp_schema_mcp.runtime.constraints import ContractCompiler
from alp_schema_mcp.runtime.protocol import ALPOptions


@pytest.fixture(scope="session")
def config():
    return ServerConfig.from_file(Path(__file__).parents[1] / "examples/catalogs.json")


@pytest.fixture(scope="session")
def compiler():
    return ContractCompiler()


@pytest.fixture(scope="session")
def profiles(config, compiler):
    catalog = config.catalogs["product-assistant"]
    return {
        op: compiler.compile(
            ALPOptions(allowed_operations=[op], catalog_ref="product-assistant"), catalog
        )
        for op in catalog.allowed_operations
    }


PAYLOADS = {
    "agent_definition_generate": {
        "resource_requirements": [],
        "requested_tools": [],
        "name": "helper_agent",
        "description": "A helper.",
        "instructions": "Answer questions.",
        "output": {"format": "text"},
    },
    "agent_call": {
        "instance_id": "product_assistant_001",
        "input": {"task": "Explain the plan.", "arguments": {}},
    },
    "agent_capability_call": {
        "instance_id": "product_assistant_001",
        "capability": "answer_product_question",
        "arguments": {"question": "How many members?"},
    },
    "list_agent_capabilities": {"instance_id": "product_assistant_001"},
    "tool_call": {"tool": "math.add", "arguments": {"a": 1, "b": 2}},
    "agent_final": {"output": {"format": "text", "value": '你好 </agent_final> " \\ 🌏'}},
}


def canonical(operation="agent_final", payload=None):
    return {
        "protocol_version": "0.3.0",
        "request_id": "req_test",
        "operation": operation,
        "payload": copy.deepcopy(PAYLOADS[operation] if payload is None else payload),
    }


def frame(operation="agent_final", payload=None):
    request = canonical(operation, payload)
    del request["operation"]
    tag = "agent_tool_call" if operation == "tool_call" else operation
    return f"<{tag}>" + json.dumps(request, ensure_ascii=False, separators=(",", ":")) + f"</{tag}>"


def body(operation="agent_final", **kwargs):
    return {
        "model": "model",
        "messages": [{"role": "user", "content": "Hello"}],
        "alp": {"allowed_operations": [operation], "catalog_ref": "product-assistant"},
        **kwargs,
    }
