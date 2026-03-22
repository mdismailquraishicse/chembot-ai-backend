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

from uuid import uuid4
from fastapi import FastAPI
from pydantic import BaseModel
from src.chembot_ai import ChatBotAI, ChatBotQuizAI, ChemDB

app = FastAPI(title="ChemBot API")
bot = ChatBotAI()
chem_db = ChemDB()
path = "data/chemistry-lr.pdf"

index_path = "faiss_index"
db = chem_db.get_faiss(path= path)

# Request schema
class QueryRequest(BaseModel):
    question: str

class QuizAnswerRequest(BaseModel):
    session_id: str
    answer: str

@app.get("/")
def home():
    return {"message": "ChemBot API is running..."}

# Response endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
   context = "\n\n".join([content.page_content for content in db.similarity_search(request.question.strip(), k=5)])
   print(f"context: {context}")
   answer = bot.ask(question=request.question.strip(), context=context)
   return {
      "answer": answer
   }

quiz_sessions = {}
@app.get("/quiz/start")
def start_quiz():
    session_id = str(uuid4())
    quiz_bot = ChatBotQuizAI()
    quizes = quiz_bot.generate_quiz()
    quiz_sessions[session_id] = quiz_bot
    return {
        "session_id":session_id,
        "question": quizes
    }

@app.post("/quiz/answer")
def answer_quiz(request: QuizAnswerRequest):
    session_id = request.session_id
    if session_id not in quiz_sessions:
        return {"error": "Invalid session_id"}
    quiz_bot = quiz_sessions[session_id]
    grade = quiz_bot.grade_answer(student_answer=request.answer)
    next_question = quiz_bot.generate_quiz()
    return {
        "result":grade,
        "next_question":next_question
    }