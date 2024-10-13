# llm_connector_factory.py
from .openai_connector import OpenAIConnector
from .anthropic_connector import AnthropicConnector
from .gemini_connector import GeminiConnector
from config import Config

def get_llm_connector(provider_name, api_key=None, model=None):
    provider_name = provider_name.lower()

    # Use provided API key if given, otherwise fetch from Config
    api_key = api_key or Config.get_api_key(provider_name)

    # Use provided model if given, otherwise fetch from Config
    model = model or Config.get_model(provider_name)

    if provider_name == "openai":
        return OpenAIConnector(api_key=api_key, model=model)
    elif provider_name == "anthropic":
        return AnthropicConnector(api_key=api_key, model=model)
    elif provider_name == "gemini":
        return GeminiConnector(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")
