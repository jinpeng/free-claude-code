"""Generic OpenAI-compatible provider implementation."""

from __future__ import annotations

from typing import Any

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.base import ProviderConfig
from providers.defaults import CUSTOM_OPENAI_DEFAULT_BASE
from providers.exceptions import InvalidRequestError
from providers.openai_compat import OpenAIChatTransport


class CustomOpenAIProvider(OpenAIChatTransport):
    """Generic provider for OpenAI-compatible ``/chat/completions`` endpoints."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CUSTOM_OPENAI",
            base_url=config.base_url or CUSTOM_OPENAI_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        try:
            return build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc
