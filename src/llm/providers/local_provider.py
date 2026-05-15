import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from src.llm.providers.base import BaseLLMProvider
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()


class LocalProvider(BaseLLMProvider):


    def __init__(self):

        hf_model = os.getenv("HF_MODEL_ID")
        local_model_id = os.getenv("LOCAL_MODEL_ID")
        path = Path(local_model_id)
        if not hf_model or not local_model_id:
            raise ValueError(f"Path not found!")
        
        if not path.exists():
            print(f"Path does not exists")
            self.download_model(hf_model, local_model_id)
        self._model, self._tokenizer = self.get_model(local_model_id = local_model_id)


    def get_llm(self):

        return self
    

    def download_model(self, hf_model:str, local_model_id:str):

        print(f"downloading and saving models...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModelForCausalLM.from_pretrained(hf_model)
        model.save_pretrained(local_model_id)
        tokenizer.save_pretrained(local_model_id)
        print(f"tokenizer and model saved successfully")


    def get_model(self, local_model_id:str):

        print(f"Loading local model")
        tokenizer = AutoTokenizer.from_pretrained(local_model_id, local_files_only = True)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_id,
            torch_dtype = "auto",
            device_map = "auto",
            local_files_only = True
            )
        return model, tokenizer


    def build_prompt(self, context:str, question:str, chat_history:list = None):


        if chat_history is None:
            chat_history = []

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
        You are ChemBot, an AI chemistry teacher.

        Rules:
        - Only answer chemistry-related questions.
        - Use the provided context to answer.
        - If the answer is not in the context, say:
            "I don't know based on the provided context."
        - If not chemistry-related, respond:
        "I can only answer chemistry-related questions."
        """),

            MessagesPlaceholder(variable_name="chat_history"),

            ("human",
            """
        Context:
        {context}

        Question:
        {question}
        """)
        ])

        natural_prompt = chat_prompt.invoke({
            "chat_history": chat_history,
            "context": context,
            "question": question
        })

        messages = natural_prompt.to_messages()
        formatted_messages = []

        role_map = {
            "system": "system",
            "human": "user",
            "ai": "assistant"
        }

        for msg in messages:
            
            formatted_messages.append(
                {
                    "role": role_map[msg.type],
                    "content": msg.content
                }
            )

        prompt = self._tokenizer.apply_chat_template(
            formatted_messages,
            tokenize = False,
            add_generation_prompt = True
        )
        print("prompt built successfully")
        print(f"built prompt: {prompt}")
        return prompt


    def invoke(self, **kwargs):


        prompt = self.build_prompt(
            context = kwargs.get("context"),
            question = kwargs.get("question"),
            chat_history = kwargs.get("chat_history")
            )

        token = self._tokenizer(
            prompt,
            return_tensors = "pt",
            truncation = True,
            max_length = 1800
            ).to(self._model.device)
        
        with torch.no_grad():

            output = self._model.generate(
                **token,
                max_new_tokens = 1024,
                temperature = 0.2,
                top_p = 0.9,
                do_sample = True
            )
        
        generated_tokens = output[0][token["input_ids"].shape[1]:] # Code to filter
        answer = self._tokenizer.decode(generated_tokens, skip_special_tokens = True)
        return AIMessage(content = answer)