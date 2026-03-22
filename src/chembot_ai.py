"""
Author: Md Ismail Quraishi
Date: 14/03/2026
Purpose:
    To implement the core AI logic for ChemBot, a chemistry-focused chatbot.
    This module handles chemistry question answering, quiz generation, and
    answer grading using a local LLM via LangChain and Ollama.
"""
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

host = os.getenv("OLLAMA_HOST")
model = os.getenv("OLLAMA_MODEL")
huggingface_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
use_local_model = os.getenv("USE_LOCAL_MODEL", "1")
pdf_path = ""
class ChemDB:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def load_pdf_from_directory(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        doc = loader.load()
        return doc
    
    def get_chunks(self, documents, chunk_size=500, chunk_overlap=50):
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Total chunks created: {len(chunks)}")
        return chunks
    
    def save_faiss(self, chunks,
                   index = "faiss_index"):
        db = FAISS.from_documents(chunks, self.embeddings)
        db.save_local(index)
        return True
    
    def load_faiss(self, index = "faiss_index"):
        db = FAISS.load_local(index, self.embeddings, allow_dangerous_deserialization=True)
        return db
    
    def get_faiss(self, path,index_path="faiss_index"):
        if os.path.exists(f"{index_path}/index.faiss"):
            print("✅ Loading existing FAISS index...")
            db = self.load_faiss(index=index_path)
        else:
            print("🚀 Creating new FAISS index...")
            document = self.load_pdf_from_directory(pdf_path=path)
            chunks = self.get_chunks(documents=document)
            self.save_faiss(index=index_path, chunks=chunks)
            db = self.load_faiss(index=index_path)
        return db
    

class ChatBotAI:
    """
    ChatBotAI handles the core conversational functionality of ChemBot.

    This class uses a local language model through LangChain and Ollama
    to answer user questions related to chemistry. It maintains a short
    conversation history to provide contextual responses while enforcing
    rules that restrict the chatbot to chemistry-related topics only.

    Responsibilities:
        - Process user chemistry questions
        - Maintain limited chat history for context
        - Generate responses using the LLM
        - Restrict responses to chemistry-related topics
    """
    def __init__(self):
        """
        Initialize the ChatBotAI instance.

        This constructor sets up the core components required for the
        ChemBot conversational system, including:

        - Initializing chat history storage to maintain recent conversation context.
        - Loading the local language model via Ollama.
        - Creating the prompt template that defines the chatbot's role,
        rules, and response behavior.

        The prompt enforces that the chatbot only answers chemistry-related
        questions and rejects non-chemistry queries.
        """

        self.chat_history = []
        if use_local_model=="1":
            # Use local model
            print(f"The program will use local model from ollama: {use_local_model}")
            self.llm  = ChatOllama(
                model = model,
                temperature = 0.0,
                host=host
            )
        else:
            # Use huggingface model
            print(f"The program will use model from huggingface: {use_local_model}")
            llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.2,
            huggingfacehub_api_token=huggingface_key
            )

            self.llm = ChatHuggingFace(llm=llm)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
        You are ChemBot, an AI chemistry teacher.

        Rules:
        - Only answer chemistry-related questions.
        - Use ONLY the provided context to answer.
        - Do NOT use your own knowledge.
        - Do NOT guess or infer beyond the context.
        - If the answer is not present in the context, respond:
        Answer not found in the provided context.
        - If the question is not chemistry-related, respond:
        I can only answer chemistry-related questions.
            """),

            MessagesPlaceholder(variable_name="chat_history"),

            ("human",
            """
        Context:
        {context}

        Question:
        {question}
        """)
        ])

        self.chain = self.prompt | self.llm

    def ask(self, question:str, context: str):
        """
        Process a user question and generate a chemistry-focused response.

        This method sends the user's question along with recent conversation
        history to the language model. The model generates a response based
        on the defined prompt rules, ensuring answers remain related to
        chemistry topics.

        The conversation history is updated after each interaction so that
        future responses can use recent context.

        Args:
            question (str): The user's input question.

        Returns:
            str: The response generated by ChemBot.
        """

        response = self.chain.invoke({
            "chat_history": self.chat_history,
            "question": question,
            "context": context
        })
        answer = response.content
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        return answer
    
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

    def __init__(self):
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
            - quiz_prompt: Prompt template used to generate chemistry quiz questions.
            - quiz_chain: LangChain pipeline that generates quiz questions.
            - grading_prompt: Prompt template used to evaluate student answers.
            - grading_chain: LangChain pipeline that grades student responses.
        """

        self.quiz_mode = False
        self.current_answer = None
        if use_local_model=="1":
            # Use local model
            print(f"The program will use local model from ollama: {use_local_model}")
            self.llm  = ChatOllama(
                model = model,
                temperature = 0.4,
                host=host
            )
        else:
            # Use huggingface model
            print(f"The program will use model from huggingface: {use_local_model}")
            llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.9,
            huggingfacehub_api_token=huggingface_key
            )

            self.llm = ChatHuggingFace(llm=llm)


        # self.llm  = ChatOllama(
        #     model = model,
        #     temperature = 0.5,
        #     host=host
        # )


        self.quiz_prompt = ChatPromptTemplate.from_template(
            """
                You are a chemistry teacher.

                Generate ONE short chemistry quiz question.
                Return ONLY in this format:

                QUESTION: <question>
                ANSWER: <correct answer>
            """
        )

        self.quiz_chain = self.quiz_prompt | self.llm
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

        result = self.quiz_chain.invoke({}).content
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        question = None
        answer = None
        for line in lines:
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
        self.current_answer = answer
        return question

    def ask(self, question: str):
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

        if question.lower() == "quiz":
            self.quiz_mode = True
            q = self.generate_quiz()
            return f"Quiz Mode Started!\n\n{q}"
        if question.lower() == "exit quiz":
            self.quiz_mode = False
            return "Quiz mode ended."
        if self.quiz_mode:
            grade = self.grade_answer(question)
            if grade.lower() == "correct":
                result = f"Correct! 🎉\n Answer:{self.current_answer}"
            else:
                result = f"Incorrect. Correct answer: {self.current_answer}"

            new_question = self.generate_quiz()

            return f"{result}\n\nNext Question:\n{new_question}"
        
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
