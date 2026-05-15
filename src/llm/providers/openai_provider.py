import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from src.llm.providers.base import BaseLLMProvider


load_dotenv()


class OpenAIProvider(BaseLLMProvider):


    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")
        if not api_key or not model:
            raise ValueError("Either OPENAI_API_KEY or OPENAI_MODEL not found!")
        self.llm = OpenAI(
            api_key = api_key,
            model = model
            )


    def get_llm(self):

        return self.llm