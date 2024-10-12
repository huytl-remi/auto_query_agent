# llm_connector_factory.py
from openai_connector import OpenAIConnector
from anthropic_connector import AnthropicConnector
from gemini_connector import GeminiConnector

def get_llm_connector(provider_name, api_key, model=None):
    provider_name = provider_name.lower()
    if provider_name == "openai":
        return OpenAIConnector(api_key=api_key, model=model or "gpt-4")
    elif provider_name == "anthropic":
        return AnthropicConnector(api_key=api_key, model=model or "claude-2")
    elif provider_name == "gemini":
        return GeminiConnector(api_key=api_key, model=model or "gemini-1.5-flash")
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")
