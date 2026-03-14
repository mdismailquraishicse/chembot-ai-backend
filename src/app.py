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

from chembot_ai import ChatBotAI, ChatBotQuizAI

bot = ChatBotAI()
quiz_bot = ChatBotQuizAI()

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    # Quiz mode
    if question.lower() == "quiz" or quiz_bot.quiz_mode:
        answer = quiz_bot.ask(question)
    # Normal mode
    else:
        answer = bot.ask(question)

    print("\nChemBot:", answer)
    print()