import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

host = os.getenv("OLLAMA_HOST")
model = os.getenv("OLLAMA_MODEL")
huggingface_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
use_local_model = os.getenv("USE_LOCAL_MODEL", "0")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# """
# 1. login system
# 2. embeddings in db
# 3. navigator in frontend
# """