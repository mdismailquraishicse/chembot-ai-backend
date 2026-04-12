"""
Author: Md Ismail Quraishi
Date: 14/03/2026

Purpose:
    This module serves as the main entry point for running the ChemBot
    command-line application.

    It initializes the chatbot components and manages user interaction
    through a terminal interface. The application supports two modes:

    1. Normal Chat Mode
       - Handled by ChatBotAI
       - Answers chemistry-related questions.

    2. Quiz Mode
       - Handled by ChatBotQuizAI
       - Generates chemistry quiz questions and evaluates user answers.

    Users can start quiz mode by typing "quiz", exit quiz mode with
    "exit quiz", and terminate the application using "exit".
"""

import os
import shutil
import asyncio
from pydantic import BaseModel
from fastapi import File, UploadFile, Request
from src.chembot_ai import  ChemDB
from src import app, use_local_model
from src.auth.models import User
from src.auth.utils import verify_password, generate_token, encrypt_password, token_validation

class Query(BaseModel):
   question:str

chem_db = ChemDB()
user_db = {}

if use_local_model == "1":
   from src.chembot_ai_with_transformers import ChatBotAI, ChatBotQuizAI
   bot = ChatBotAI()
   quiz_bot = ChatBotQuizAI(model=bot.model,
                            tokenizer=bot.tokenizer)
else:
   from src.chembot_ai import ChatBotAI, ChatBotQuizAI
   bot = ChatBotAI()
   quiz_bot = ChatBotQuizAI(flag_quiz=True)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/register")
def register(user:User):
   """
   1. generate hashed password
   2. store creds in db
   """
   password = user.password
   password_hash = encrypt_password(password=password)
   user.password = password_hash
   user_db[user.email] = user.model_dump()
   print(f"registered user: {user_db}")
   return True

@app.post("/login")
def login(user:User):
   """
      1. Fetch hashed password from db for the given username
      2. compare hashed and plain password using bcrypt
   """
   # fetch plain password from db
   print(f"user:{user}")
   print(f"user db : {user_db}")
   hash_pw = user_db.get(user.email).get("password")
   print(f"user: {user}")
   if not verify_password(hash_pw=hash_pw ,password=user.password):
      print(f"Invalid credentials")
      return
   token = generate_token(payload=user.model_dump())
   return token


@app.get("/")
@token_validation
def home():
   return {
      "message":"chembot is running..."
   }

@app.post("/upload-pdf")
# @token_validation
async def upload_pdf(file:UploadFile=File(...)):
   if file.content_type != "application/pdf":
      return {"error": "only pdf files are allowed"}
   file_path = os.path.join(UPLOAD_DIR, file.filename)
   with open(file_path, "wb") as buffer:
      shutil.copyfileobj(file.file, buffer)
   print(f"file uploaded successfully")
   bot.filename = file.filename
   quiz_bot.filename = file.filename   
   print(f"bot filename: {bot.filename}")
   print(f"quiz bot filename: {quiz_bot.filename}")
   
   return {
      "filename":file.filename,
      "message": "pdf uploaded successfully"
   }

@app.post("/ask")
# @token_validation
async def ask(query: Query, request:Request):
   print(f"ask is called")
   file_path = os.path.join(UPLOAD_DIR, bot.filename)
   db = chem_db.get_faiss(path=file_path)
   question = query.question.strip()
   context = "\n\n".join([content.page_content for content in db.similarity_search(question.strip(), k=2)])
   if use_local_model == "1":
      print(f"using local model")
      answer = await asyncio.to_thread(
         bot.invoke_local_model,
         user_input=question,
         context=context
      )
   else:
      print(f"using huggingface hosted model")
      print(f"context: {context}")
      bot.flag = False
      answer = bot.ask(question=question, context=context)
   return {
      "answer":answer
   }


@app.post("/quiz")
@token_validation
async def quiz(query: Query, request:Request):
   quiz_que, options, quiz_answer = await asyncio.to_thread(
      quiz_bot.ask,
      question = query.question)
   

   print(f"response: {quiz_que}")
   print(f"answer: {quiz_answer}")
   return {"quiz":quiz_que,
           "answer": quiz_answer,
           "options":options
           }

@app.get("/quiz/{answer}")
@token_validation
def quiz_answer(answer:int, request:Request):
   if answer==1:
      print("the answer is correct")
      return {"result": "Correct"}
   else:
      print("the answer is incorrect")
      return {"result": "Incorrect"}
