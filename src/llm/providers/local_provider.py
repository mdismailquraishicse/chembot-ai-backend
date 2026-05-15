import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from src.llm.providers.base import BaseLLMProvider
from transformers import AutoTokenizer, AutoModelForCausalLM


load_dotenv()


class LocalProvider(BaseLLMProvider):


    def __init__(self):

        hf_model = os.getenv("HF_MODEL_ID")
        local_model_id = os.getenv("LOCAL_MODEL_ID")
        path = Path(local_model_id)
        if not hf_model and not local_model_id:
            raise ValueError(f"Path not found!")
        
        if not path.exists():
            print(f"Path does not exists")
            self.download_model(hf_model, local_model_id)
        self._model, self._tokenizer = self.get_model(local_model_id = local_model_id)


    def get_llm(self):

        return self.llm
    

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
    

    def invoke(self, prompt):

        token = self._tokenizer(
            prompt,
            return_tensors = "pt",
            truncate = True,
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
        return {"content":answer}