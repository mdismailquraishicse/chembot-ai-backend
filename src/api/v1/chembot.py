import os
from fastapi import APIRouter, Request
from src.utils.util import token_validation
from src.pydantic_models.chembot import Query
from src.services.chembot_ai import ChatBotAI
from src.services.chembot_quiz import ChatBotQuizAI


router = APIRouter()
provider = os.getenv("PROVIDER", "hf")

chembot = ChatBotAI(provider = provider)
quizbot = ChatBotQuizAI( provider = provider)


@router.post("/ask")
@token_validation
async def ask(query: Query, request:Request):

   print(f"ask is called")
   question = query.question.strip()
   if provider.lower() == "local":
      answer = await chembot.ask_local(question=question, local_provider = True)
   else:
      answer = chembot.ask(question=question)
   return {
   "answer":answer
   }


@router.post("/quiz")
@token_validation
async def quiz(request:Request):

    quiz_que, options, quiz_answer = quizbot.invoke()
    print(f"response: {quiz_que}")
    print(f"answer: {quiz_answer}")
    return {"quiz":quiz_que,
           "answer": quiz_answer,
           "options":options
           }


@router.get("/quiz/{answer}")
@token_validation
def quiz_answer(answer:int, request:Request):

    if answer==1:
        print("the answer is correct")
        return {"result": "Correct"}
    else:
        print("the answer is incorrect")
        return {"result": "Incorrect"}
