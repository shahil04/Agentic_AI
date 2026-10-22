"""Provider configuration and LangChain client factories for the AI applications."""

from dataclasses import dataclass
import os


_CHAT_DEFAULTS = {
    "gemini": ("GEMINI_CHAT_MODEL", "gemini-3.5-flash"),
    "groq": ("GROQ_CHAT_MODEL", "meta-llama/llama-prompt-guard-2-22m"),
    "openai": ("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_CHAT_MODEL", "claude-3-5-haiku-latest"),
}
_CHAT_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_EMBEDDING_DEFAULTS = {
    "gemini": ("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    "openai": ("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
}


@dataclass(frozen=True)
class AISettings:
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str

    @property
    def model_identifier(self) -> str:
        return f"{self.llm_provider}/{self.llm_model}"


def _configured_provider(variable: str, allowed: dict[str, tuple[str, str]]) -> str:
    provider = os.getenv(variable, "gemini").strip().lower()
    if provider not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"{variable} must be one of: {choices}; received {provider!r}.")
    return provider


def load_settings() -> AISettings:
    llm_provider = _configured_provider("LLM_PROVIDER", _CHAT_DEFAULTS)
    embedding_provider = _configured_provider("EMBEDDING_PROVIDER", _EMBEDDING_DEFAULTS)
    chat_variable, default_chat_model = _CHAT_DEFAULTS[llm_provider]
    embedding_variable, default_embedding_model = _EMBEDDING_DEFAULTS[embedding_provider]
    llm_model = os.getenv(chat_variable, "").strip() or default_chat_model
    embedding_model = os.getenv(embedding_variable, "").strip() or default_embedding_model

    return AISettings(
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )


def _required_api_key(variable: str) -> str:
    value = os.getenv(variable, "").strip()
    if not value:
        raise ValueError(f"Set {variable} to use the configured AI provider.")
    return value


def create_chat_model(temperature: float = 0.0, settings: AISettings | None = None):
    settings = settings or load_settings()
    api_key = _required_api_key(_CHAT_KEYS[settings.llm_provider])

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=api_key,
            temperature=temperature,
        )
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.llm_model, api_key=api_key, temperature=temperature)
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model, api_key=api_key, temperature=temperature)
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=api_key,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider!r}.")


def create_embedding_model(settings: AISettings | None = None):
    settings = settings or load_settings()
    if settings.embedding_provider == "gemini":
        api_key = _required_api_key("GEMINI_API_KEY")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=api_key,
        )
    if settings.embedding_provider == "openai":
        api_key = _required_api_key("OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, api_key=api_key)
    raise ValueError(
        "EMBEDDING_PROVIDER must be 'gemini' or 'openai'; "
        f"received {settings.embedding_provider!r}."
    )
