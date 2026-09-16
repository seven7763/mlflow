from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig, _OpenAICompatibleConfig
from mlflow.gateway.providers.daoxe import DaoxeProvider
from mlflow.gateway.schemas import chat

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider(provider_config=None, model_name="claude-sonnet-4") -> DaoxeProvider:
    endpoint_config = EndpointConfig(
        name="daoxe-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "daoxe",
            "name": model_name,
            "config": {"api_key": "sk-daoxe-test-key", **(provider_config or {})},
        },
    )
    return DaoxeProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-daoxe-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "claude-sonnet-4",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from DaoXE!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "DaoXE"


def test_config_type():
    provider = _make_provider()
    assert isinstance(provider.config.model.config, _OpenAICompatibleConfig)


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "https://api.daoxe.com/v1"


def test_api_base_override():
    provider = _make_provider({"api_base": "https://daoxe.internal.example.com/v1"})
    assert provider._api_base == "https://daoxe.internal.example.com/v1"


def test_headers():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer sk-daoxe-test-key"}


@pytest.mark.asyncio
async def test_chat():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session:
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["id"] == "chatcmpl-daoxe-123"
    assert result["choices"][0]["message"]["content"] == "Hello from DaoXE!"

    session_headers = mock_session.call_args.kwargs["headers"]
    assert session_headers["Authorization"] == "Bearer sk-daoxe-test-key"
    mock_client.post.assert_called_once()
    assert mock_client.post.call_args.args[0] == "https://api.daoxe.com/v1/chat/completions"
