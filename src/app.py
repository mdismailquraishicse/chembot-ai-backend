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
from src import app
from fastapi import Request
from pydantic import BaseModel
from src.auth.models import User
from src.services.chembot_ai import ChatBotAI
from src.services.chembot_quiz import ChatBotQuizAI
from src.auth.utils import verify_password, generate_token, encrypt_password, token_validation

class Query(BaseModel):
   question:str

provider = os.getenv("PROVIDER", "hf")
print(f"MODEL PROVIDER: {provider}")
chembot = ChatBotAI(provider = provider)
quiz_bot = ChatBotQuizAI(provider = provider)
user_db = {}


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
def home():

   return {
      "message":"chembot is running..."
   }


@app.post("/ask")
@token_validation
async def ask(query: Query, request:Request):

    print(f"ask is called")
    question = query.question.strip()
    answer =await chembot.ask(question=question)
    return {
    "answer":answer
    }


@app.post("/quiz")
@token_validation
async def quiz(request:Request):

    quiz_que, options, quiz_answer = quiz_bot.invoke()
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

