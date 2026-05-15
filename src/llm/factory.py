from src.llm.providers.local_provider import LocalProvider
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.providers.huggingface_provider import HuggingFaceProvider



class LLMFactory:


    @staticmethod
    def create(provider:str):

        providers = {
            "hf": HuggingFaceProvider,
            "openai": OpenAIProvider,
            "local": LocalProvider
        }

        if provider not in providers:
            raise ValueError("Unsupported provider")
        
        return providers[provider]().get_llm()