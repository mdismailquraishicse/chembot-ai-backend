import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class ChatBotAI:
    def __init__(self):
        self.llm  = ChatOllama(
            model = "gemma:2b",
            temperature = 0.5
        )

        self.chat_history = []

        self.prompt = ChatPromptTemplate.from_template(
            """
                You are ChemBot, an AI chemistry teacher.

                Rules:
                - Only answer questions related to chemistry.
                - If the question is not about chemistry, respond with:
                "I can only answer chemistry-related questions."

                Conversation history:
                {history}

                User question:
                {question}
            """
        )

        self.chain = self.prompt | self.llm

    def ask(self, question:str):
        history_text = "\n".join(self.chat_history)
        response = self.chain.invoke({
            "history": history_text,
            "question": question
        })
        answer = response.content
        self.chat_history.append(f"User: {question}")
        self.chat_history.append(f"ChemBot: {answer}")
        return answer