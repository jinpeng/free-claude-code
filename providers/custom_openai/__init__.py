"""Generic OpenAI-compatible custom provider exports."""

from providers.defaults import CUSTOM_OPENAI_DEFAULT_BASE

from .client import CustomOpenAIProvider

__all__ = [
    "CUSTOM_OPENAI_DEFAULT_BASE",
    "CustomOpenAIProvider",
]
