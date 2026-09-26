from langchain_core.language_models import BaseChatModel

from .config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Create the configured provider without coupling agents to an SDK."""
    provider = settings.model_provider
    model = settings.model_name
    api_key = settings.model_api_key()

    if provider in {"openai", "openai-compatible"}:
        from langchain_openai import ChatOpenAI

        kwargs = {"model": model, "api_key": api_key}
        if settings.model_base_url:
            kwargs["base_url"] = settings.model_base_url
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=api_key)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=model, api_key=api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_base_url)

    raise ValueError(f"Unsupported model provider: {provider}")
