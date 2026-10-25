"""Chat models through ``init_chat_model``: local ollama by default, Anthropic optional,
a fake for offline runs."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from tag_agent.config import Settings
from tag_agent.fake import FakeTagModel

__all__ = ["ModelConfigError", "is_loopback_url", "make_model"]


class ModelConfigError(RuntimeError):
    """The configured provider can't be used as configured."""


def is_loopback_url(url: str) -> bool:
    host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_model(settings: Settings) -> BaseChatModel:
    if settings.provider == "fake":
        return FakeTagModel()
    if settings.provider == "ollama":
        if not settings.allow_remote_ollama and not is_loopback_url(settings.ollama_host):
            raise ModelConfigError(
                f"OLLAMA_HOST={settings.ollama_host!r} is not loopback; set "
                "TAG_ALLOW_REMOTE_OLLAMA=1 to send prompts (and table rows) to another machine"
            )
        model = init_chat_model(
            settings.model,
            model_provider="ollama",
            base_url=settings.ollama_host,
            temperature=settings.temperature,
            reasoning=False,
        )
    elif settings.provider == "anthropic":  # pragma: no cover - optional extra + a key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelConfigError("ANTHROPIC_API_KEY is not set")
        try:
            model = init_chat_model(
                settings.model, model_provider="anthropic", temperature=settings.temperature
            )
        except ImportError as exc:
            raise ModelConfigError(
                "pip install 'langchain-tag-agent[anthropic]' to use Anthropic"
            ) from exc
    else:
        raise ModelConfigError(f"unknown provider {settings.provider!r}")
    if not isinstance(model, BaseChatModel):  # pragma: no cover - init_chat_model contract
        raise ModelConfigError("init_chat_model did not return a chat model")
    return model
