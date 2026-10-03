from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Operation = Literal[
    "agent_definition_generate",
    "agent_call",
    "agent_capability_call",
    "list_agent_capabilities",
    "tool_call",
    "agent_final",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OperationChoice(StrictModel):
    operation: Operation


class ALPOptions(StrictModel):
    protocol_version: Literal["0.3.0", "0.4.0"] = "0.3.0"
    allowed_operations: list[Operation] = Field(min_length=1, max_length=6)
    choice: Literal["required"] | OperationChoice = "required"
    strict: Literal[True] = True
    catalog_ref: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def check_operations(self):
        if len(set(self.allowed_operations)) != len(self.allowed_operations):
            raise ValueError("allowed_operations must be unique")
        if isinstance(self.choice, OperationChoice) and self.choice.operation not in self.allowed_operations:
            raise ValueError("The selected operation must be allowed")
        return self


class AgentCall(StrictModel):
    id: str
    type: Literal["alp"] = "alp"
    request: dict[str, Any]


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    agent_calls: list[AgentCall] | None = Field(default=None, min_length=1, max_length=16)
    name: str | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def check_content(self):
        if self.agent_calls is not None:
            if self.role != "assistant" or self.content is not None:
                raise ValueError("agent_calls requires assistant role and null content")
        elif self.content is None:
            raise ValueError("A message requires content or agent_calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("A tool message requires tool_call_id")
        return self


class ALPChatRequest(StrictModel):
    model: str = Field(min_length=1, max_length=512)
    messages: list[Message] = Field(min_length=1, max_length=256)
    alp: ALPOptions
    stream: bool = False
    n: Literal[1] = 1
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=-(2**63), lt=2**63)
    include_raw: bool = False


    @model_validator(mode="after")
    def check_history_version(self):
        for message in self.messages:
            if message.agent_calls:
                if self.alp.protocol_version == "0.3.0" and len(message.agent_calls) != 1:
                    raise ValueError("ALP 0.3 requires exactly one call per turn")
                if len({call.id for call in message.agent_calls}) != len(message.agent_calls):
                    raise ValueError("Assistant call IDs must be unique")
        return self


class AssistantMessage(StrictModel):
    role: Literal["assistant"] = "assistant"
    content: None = None
    agent_calls: list[AgentCall] = Field(min_length=1, max_length=16)


class Choice(StrictModel):
    index: Literal[0] = 0
    message: AssistantMessage
    finish_reason: Literal["agent_calls"] = "agent_calls"


class ALPChatResponse(StrictModel):
    id: str
    object: Literal["alp.chat.completion"] = "alp.chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: dict[str, Any] | None = None
    alp: dict[str, Any]
    raw: str | None = None
