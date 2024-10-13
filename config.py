# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    DEFAULT_LLM_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3.5-sonnet')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')

    @classmethod
    def get_api_key(cls, provider):
        return getattr(cls, f"{provider.upper()}_API_KEY")

    @classmethod
    def get_model(cls, provider):
        return getattr(cls, f"{provider.upper()}_MODEL")
