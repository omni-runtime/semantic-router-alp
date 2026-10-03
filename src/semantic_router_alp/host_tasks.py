"""Typed host task inputs compiled to existing, request-signed ALP constraints.

These values come from the caller's task contract, never from model text or a
conformance oracle. This module does not authorize or execute the resulting action.
"""
from __future__ import annotations

import copy
from typing import Annotated, Any, Literal

from alp_schema_mcp.catalog import ContractCatalog
from jsonschema import Draft202012Validator
from pydantic import Field, TypeAdapter, model_validator

from .catalog import CapabilityContract, PayloadConstraints, ResponseConstraints
from .protocol import ALPChatRequest, StrictModel


class AgentCallTask(StrictModel):
    operation: Literal["agent_call"] = "agent_call"
    instance_id: str
    task: str = Field(min_length=1, max_length=32000)
    session_mode: Literal["continue", "isolated"]
    artifacts: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    arguments: dict[str, Any] | None = None
    expected_state_version: int | None = Field(default=None, ge=1, le=2147483647)

    def payload(self):
        result = {
            "instance_id": self.instance_id,
            "input": {"task": self.task, "artifacts": copy.deepcopy(self.artifacts)},
            "session_mode": self.session_mode,
        }
        if self.arguments is not None:
            result["input"]["arguments"] = copy.deepcopy(self.arguments)
        if self.expected_state_version is not None:
            result["expected_state_version"] = self.expected_state_version
        return result

    @model_validator(mode="after")
    def validate_protocol_fields(self):
        schema = ContractCatalog().schema("protocol", "AgentCall")
        if not Draft202012Validator(schema).is_valid(self.payload()):
            raise ValueError("Host call task must satisfy the ALP AgentCall contract")
        return self

    def constraints(self):
        payload = self.payload()
        fixed = {"/" + k: v for k, v in payload.items() if k != "input"}
        fixed.update({"/input/" + k: v for k, v in payload["input"].items()})
        return PayloadConstraints(fixed_values=fixed, forbidden_fields=[] if self.expected_state_version is not None else ["/expected_state_version"])


class DefinitionTask(StrictModel):
    operation: Literal["agent_definition_generate"] = "agent_definition_generate"
    name: str | None = None
    capabilities: dict[str, CapabilityContract] = Field(default_factory=dict, max_length=32)
    output_from_capability: str | None = None
    output: dict[str, Any] | None = None
    environment_profile_ref: str | None = None
    requested_tools: list[str] | None = None
    resource_requirements: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_contracts(self):
        PayloadConstraints(capabilities=self.capabilities)
        if not self.capabilities and all(getattr(self, k) is None for k in (
            "name", "output", "environment_profile_ref", "requested_tools", "resource_requirements",
        )):
            raise ValueError("A host definition task must bind at least one requirement")
        contracts = ContractCatalog()
        for field, definition in (("name", "Name"), ("output", "OutputContract"),
                                  ("environment_profile_ref", "CatalogRef")):
            value = getattr(self, field)
            if value is not None and not Draft202012Validator(contracts.schema("protocol", definition)).is_valid(value):
                raise ValueError(f"Host {field} must satisfy the ALP contract")
        if self.output_from_capability is not None:
            if self.output is not None:
                raise ValueError("Select output or output_from_capability, not two output sources")
            cap = self.capabilities.get(self.output_from_capability)
            if cap is None or cap.output_schema is None:
                raise ValueError("output_from_capability must reference a supplied output schema")
        return self

    def constraints(self):
        fixed = {}
        for name in ("name", "requested_tools", "resource_requirements", "output", "environment_profile_ref"):
            value = getattr(self, name)
            if value is not None:
                fixed["/" + name] = copy.deepcopy(value)
        if self.output_from_capability is not None:
            fixed["/output"] = {
                "format": "json",
                "schema": copy.deepcopy(self.capabilities[self.output_from_capability].output_schema),
            }
        return PayloadConstraints(fixed_values=fixed, capabilities=copy.deepcopy(self.capabilities))


HostTask = Annotated[AgentCallTask | DefinitionTask, Field(discriminator="operation")]
HOST_TASK = TypeAdapter(HostTask)


def host_task_headers(request: ALPChatRequest, task: HostTask | dict, *, key: bytes,
                      ttl_seconds: int = 60, now: int | None = None):
    """The host SDK entry point, shared by vLLM, Omni and MLX endpoints."""
    from .task_context import task_headers

    task = HOST_TASK.validate_python(task)
    if task.operation not in request.alp.allowed_operations:
        raise ValueError("Host task operation is not allowed by this request")
    choice = request.alp.choice
    if choice != "required" and choice.operation != task.operation:
        raise ValueError("Host task operation conflicts with the selected operation")
    # A typed single task must not leave a different operation available as an
    # unconstrained escape route. Callers select the operation in the request.
    if choice == "required" and request.alp.allowed_operations != [task.operation]:
        raise ValueError("Select the host task operation explicitly")
    constraints = task.constraints()
    response = ResponseConstraints(members=[{"operation": task.operation, "payload": constraints}]) if request.alp.protocol_version == "0.4.0" else None
    return task_headers(request, {} if response else {task.operation: constraints}, key=key,
                        response=response, ttl_seconds=ttl_seconds, now=now)


def host_response_headers(request, members, *, key, ttl_seconds=60, now=None):
    """Sign reviewed ordered operation/payload requirements for one 0.4 turn."""
    from .task_context import task_headers
    if request.alp.protocol_version != "0.4.0":
        raise ValueError("An ordered host response requires ALP 0.4")
    plan = ResponseConstraints(members=members)
    if any(m.operation not in request.alp.allowed_operations for m in plan.members):
        raise ValueError("Host response operation is unavailable")
    return task_headers(request, {}, response=plan, key=key, ttl_seconds=ttl_seconds, now=now)
