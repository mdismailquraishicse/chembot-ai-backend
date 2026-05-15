"""
Author: Md Ismail Quraishi
Date: 15/05/2026
Purpose:
    To implement the core AI logic for ChemBot, a chemistry-focused chatbot.
    This module handles chemistry question answering, quiz generation, and
    answer grading using a local LLM via LangChain and Ollama.
"""

import os
import random
from dotenv import load_dotenv
from src.db.chembot_ai import ChemDB
from src.llm.factory import LLMFactory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chem_db = ChemDB()
load_dotenv()

# prompt_quiz = ChatPromptTemplate.from_template(
#             """
#                 You are a chemistry teacher.

#                 Generate ONE short chemistry quiz question.
#                 Return ONLY in this format:

#                 QUESTION: <question>
#                 ANSWER: <correct answer>
#                 OPTION_A: <incorrect answer>
#                 OPTION_B: <incorrect answer>
#                 OPTION_C: <incorrect answer>
#             """
#         )

prompt_quiz = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a chemistry teacher.
        Generate EXACTLY ONE chemistry quiz question.
        
        Rules:
        - Question should be short.
        - Provide 1 correct answer and 3 incorrect options.
        - Do not repeat previous questions from chat history.
        - Keep difficulty medium.
        - Return ONLY in this format:
        
        QUESTION: <question>
        ANSWER: <correct answer>
        OPTION_A: <incorrect answer>
        OPTION_B: <incorrect answer>
        OPTION_C: <incorrect answer>
        """
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        "Generate a new chemistry quiz question."
    )
])

    
class ChatBotQuizAI:

    """
    Quiz engine for ChemBot that manages chemistry quiz interactions.

    This class is responsible for generating chemistry quiz questions,
    evaluating student answers, and controlling the quiz flow. It uses
    a local language model via LangChain and Ollama to generate questions
    and optionally assist in grading responses.

    The class maintains quiz state (quiz mode and current correct answer)
    and supports starting, continuing, and exiting quiz sessions.

    Responsibilities:
        - Generate chemistry quiz questions
        - Store the correct answer for each question
        - Grade student responses
        - Manage quiz session state
    """

    def __init__(self, provider):

        """
        Initialize the quiz engine for ChemBot.

        This constructor prepares the components required for running
        chemistry quiz sessions. It initializes the quiz state, loads
        the local language model through Ollama, and defines the prompt
        templates used for generating quiz questions and grading
        student answers.

        Key initializations:
            - quiz_mode: Tracks whether the quiz session is active.
            - current_answer: Stores the correct answer for the current quiz question.
            - llm: Loads the local language model used for quiz generation and grading.
            - prompt: Prompt template used to generate chemistry quiz questions.
            - quiz_chain: LangChain pipeline that generates quiz questions.
            - grading_prompt: Prompt template used to evaluate student answers.
            - grading_chain: LangChain pipeline that grades student responses.
        """

        self.chat_history = []
        self.current_answer = None
        self.llm = LLMFactory.create(provider=provider)
        self.prompt = prompt_quiz
        self.chain = self.prompt | self.llm

        self.grading_prompt = ChatPromptTemplate.from_template(
            """
            You are a chemistry teacher grading a student's answer.

            Correct answer:
            {correct_answer}

            Student answer:
            {student_answer}

            Rules:
            - Ignore capitalization differences.
            - Ignore extra explanatory words.
            - If the student's answer contains the correct concept, mark it CORRECT.
            - Only mark INCORRECT if the scientific concept is wrong.

            Respond with ONLY one word:
            CORRECT
            or
            INCORRECT
            """
            )
        
        self.grading_chain = self.grading_prompt | self.llm

    def generate_quiz(self):

        """
        Generate a new chemistry quiz question using the language model.

        This method invokes the quiz generation chain to create a quiz
        question and its corresponding correct answer. The model is
        expected to return the result in a structured format containing
        'QUESTION:' and 'ANSWER:' labels.

        The method parses the model output to extract the quiz question
        and the correct answer. The correct answer is stored internally
        for later evaluation when the student submits their response.

        Returns:
            str: The generated chemistry quiz question.
        """

        result = self.chain.invoke({
            "chat_history": self.chat_history
        }).content
        self.chat_history.append(result)
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        question = None
        answer = None

        for line in lines:
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("OPTION_A:"):
                opt_a = line.replace("OPTION_A:", "").strip()
            elif line.startswith("OPTION_B:"):
                opt_b = line.replace("OPTION_B:", "").strip()
            elif line.startswith("OPTION_C:"):
                opt_c = line.replace("OPTION_C:", "").strip()
        self.current_answer = answer
        option_keys = ["a", "b", "c", "d"]
        all_options = [opt_a, opt_b, opt_c, answer]
        all_options = list(set(all_options))
        print(f"before shuffle all options: {all_options}")
        random.shuffle(all_options)
        print(f"all options: {all_options}")
        self.options = {key:val for key, val in zip(option_keys, all_options)}
        return question

    def invoke(self):

        """
        Handle user input during a quiz session.

        This method controls the quiz workflow including starting a quiz,
        exiting quiz mode, grading student answers, and generating the
        next quiz question.

        Behavior:
            - If the user enters "quiz", quiz mode is activated and a new
            chemistry question is generated.
            - If the user enters "exit quiz", the quiz session is terminated.
            - If quiz mode is active, the student's answer is graded using
            the grading mechanism, feedback is provided, and a new quiz
            question is generated.

        Args:
            question (str): The user's input, which may be a command
                            (e.g., "quiz", "exit quiz") or an answer to
                            the current quiz question.

        Returns:
            str: Feedback on the student's answer along with the next
                quiz question or system message.
        """

        q = self.generate_quiz()
        print(f"generated quiz q: {q}")
        print(f"answer: {self.current_answer}")
        print(f"options: {self.options}")
        return q, self.options, self.current_answer
        
    def grade_answer(self, student_answer: str):

        """
        Evaluate a student's quiz answer using the grading LLM chain.

        This method sends the correct answer and the student's response
        to the grading prompt, which determines whether the student's
        answer is scientifically correct based on the defined grading rules.
        The model returns either "CORRECT" or "INCORRECT".

        Args:
            student_answer (str): The answer provided by the student.

        Returns:
            str: The grading result returned by the model ("CORRECT" or "INCORRECT").
        """

        result = self.grading_chain.invoke({
            "correct_answer": self.current_answer,
            "student_answer": student_answer
        }).content.strip()
        return result

