from src.llm.providers.huggingface_provider import HuggingFaceProvider



class LLMFactory:


    @staticmethod
    def create(provider:str):

        providers = {
            "hf": HuggingFaceProvider,
            "local": None
        }

        if provider not in providers:
            raise ValueError("Unsupported provider")
        
        return providers[provider]().get_llm()