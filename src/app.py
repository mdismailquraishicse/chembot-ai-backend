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

from src.chembot_ai import ChatBotAI, ChatBotQuizAI, ChemDB

bot = ChatBotAI()
quiz_bot = ChatBotQuizAI()
chem_db = ChemDB()
path = "src/chemistry-lr.pdf"

index_path = "faiss_index"
db = chem_db.get_faiss(path= path)

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    # Quiz mode
    if question.lower() == "quiz" or quiz_bot.quiz_mode:
        answer = quiz_bot.ask(question)
    # Normal mode
    else:
      docs = db.similarity_search_with_score(question.strip(), k=5)
   #    filtered_docs = [
   #       doc for doc, score in docs if score < 0.8
   #    ]
   #    if not filtered_docs:
   #       context = ""
   #    else:
   #       context = "\n\n".join([doc.page_content for doc in filtered_docs])
      context = "\n\n".join([content.page_content for content in db.similarity_search(question.strip(), k=2)])
      print(f"context: {context}")
      answer = bot.ask(question, context=context)

    print("\nChemBot:", answer)
    print()