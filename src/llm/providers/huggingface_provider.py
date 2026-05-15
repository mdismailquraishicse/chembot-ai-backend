import os
from dotenv import load_dotenv
from src.llm.providers.base import BaseLLMProvider
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()


class HuggingFaceProvider(BaseLLMProvider):

    def __init__(self):

        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        repo_id = os.getenv("HF_REPO_ID")
        if not api_key or not repo_id:
            print(f"api_key: {api_key} repo_id: {repo_id}")
            raise ValueError("Either HUGGINGFACEHUB_API_TOKEN or HF_REPO_ID not found!")

        endpoint = HuggingFaceEndpoint(
            huggingfacehub_api_token = api_key,
            repo_id = repo_id,
            temperature = 0.5,
            max_new_tokens = 1024
        )
        self.llm = ChatHuggingFace(llm = endpoint)


    def get_llm(self):

        return self.llm
