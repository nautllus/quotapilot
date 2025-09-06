import os
import sys
import time

# Ensure 'src' is on the import path for tests
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from gateway.schemas import (
    ChatMessage,
    ChatToolCall,
    ChatRequest,
    ChatResponse,
    ChatChoice,
)


def test_chat_message_optional_fields():
    msg = ChatMessage(
        role="assistant",
        content=None,  # content can be None for tool responses
        tool_calls=[ChatToolCall(function={"name": "do_something", "arguments": "{}"})],
    )
    assert msg.content is None
    assert msg.tool_calls is not None and len(msg.tool_calls) == 1


def test_chat_request_optional_params():
    req = ChatRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.7,
        max_tokens=128,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.1,
        seed=42,
        response_format={"type": "text"},
        n=1,
        stop=["\n\n"],
        logprobs=False,
        top_logprobs=3,
    )
    assert req.temperature == 0.7
    assert req.n == 1


def test_chat_response_structure():
    msg = ChatMessage(role="assistant", content="ok")
    choice = ChatChoice(index=0, message=msg, finish_reason="stop")
    resp = ChatResponse(id="test", created=int(time.time()), model="auto", choices=[choice])
    assert resp.object == "chat.completion"
    assert resp.usage.total_tokens == 0

