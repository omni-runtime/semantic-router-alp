import pytest
from conftest import frame

from semantic_router_alp.errors import ALPError
from semantic_router_alp.parser import ALPParser


@pytest.mark.parametrize(
    "transform",
    [
        lambda text: text + "trailing",
        lambda text: text + text,
        lambda text: text.replace(
            '"request_id":"req_test"', '"request_id":"req_test","request_id":"duplicate"'
        ),
        lambda text: text.replace('"protocol_version":"0.3.0"', '"protocol_version":"0.2.0"'),
        lambda text: text[:-5],
    ],
)
def test_full_validator_rejects_invalid_output(transform, compiler, profiles):
    parser = ALPParser(profiles["agent_final"], compiler.contracts)
    parser.feed(transform(frame()))
    with pytest.raises(ALPError):
        parser.finish("stop")


@pytest.mark.parametrize("reason", ["length", "abort", "tool_calls", None])
def test_even_complete_frame_is_not_accepted_after_abnormal_end(reason, compiler, profiles):
    parser = ALPParser(profiles["agent_final"], compiler.contracts)
    parser.feed(frame())
    with pytest.raises(ALPError) as exc:
        parser.finish(reason)
    assert exc.value.code == "INCOMPLETE_GENERATION"


def test_feed_and_finish_after_completion_fail(compiler, profiles):
    parser = ALPParser(profiles["agent_final"], compiler.contracts)
    parser.feed(frame())
    parser.finish("stop")
    with pytest.raises(ALPError):
        parser.feed(" ")
    with pytest.raises(ALPError):
        parser.finish("stop")


def test_bounded_utf8_and_depth(compiler, profiles):
    parser = ALPParser(profiles["agent_final"], compiler.contracts)
    with pytest.raises(ALPError) as exc:
        parser.feed("你" * 50000)
    assert exc.value.code == "OUTPUT_TOO_LARGE"
    parser = ALPParser(profiles["agent_final"], compiler.contracts)
    with pytest.raises(ALPError) as exc:
        parser.feed("[" * 33)
    assert exc.value.code == "OUTPUT_TOO_DEEP"


def test_unexpected_operation(compiler, profiles):
    parser = ALPParser(profiles["agent_final"], compiler.contracts)
    parser.feed(frame("agent_call"))
    with pytest.raises(ALPError) as exc:
        parser.finish("stop")
    assert exc.value.code == "UNEXPECTED_OPERATION"
