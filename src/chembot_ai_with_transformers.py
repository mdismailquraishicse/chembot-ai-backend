import os
import time
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

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

    def __init__(self, model=None, tokenizer=None):
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
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
            return

        self.model_id = os.getenv("model_id")
        self.model_id_local = os.getenv("model_id_local")
        path = Path(self.model_id_local)
        if not path.exists():
            print("directory not exist")
            self.download_model()
        self.model, self.tokenizer = self.get_model()       
        print(f"path: {path}")

    def download_model(self):
        print(f"model {self.model_id} is being loaded...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.save_pretrained(self.model_id_local)
        tokenizer.save_pretrained(self.model_id_local)
        print(f"model {self.model_id} loaded and saved successfully to {self.model_id_local}")

    def get_model(self):
        print("loading local model...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id_local, local_files_only=True)
        print("tokenizer loaded successfully")
        model = AutoModelForCausalLM.from_pretrained(
        self.model_id_local,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True
        )
        print("model loaded successfully")
        return model, tokenizer
    
    def prompt_builder_with_langchain(self, context, user_input):
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
        You are ChemBot, an AI chemistry teacher.

        Rules:
        - Only answer chemistry-related questions.
        - Use the provided context to answer.
        - If the answer is not in the context, say:
            "I don't know based on the provided context."
        - If not chemistry-related, respond:
        "I can only answer chemistry-related questions."
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
        natural_prompt = chat_prompt.invoke({
            "chat_history":self.chat_history,
            "context":context,
            "question":user_input
        })

        prompt = natural_prompt.to_string()
        print("prompt built successfully")
        print(f"built prompt: {prompt}")
        return prompt

    def invoke_local_model(self, user_input, context):
        init_time = time.time()
        MAX_HISTORY = 6
        print(f"input : {user_input}")

        prompt = self.prompt_builder_with_langchain(context=context,
                                                   user_input=user_input)
        
        token = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(self.model.device)
        
        with torch.no_grad():
            outputs =self.model.generate(**token,
                                max_new_tokens=100,
                                temperature=0.2,
                                top_p=0.9,
                                do_sample=True)
        generated_tokens = outputs[0][token["input_ids"].shape[1]:] # This line will filter only answer from prompt+answer
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=answer))
        self.chat_history = self.chat_history[-MAX_HISTORY:]
        print(f"total time taken: {time.time()-init_time}")
        return answer
    
    def invoke_hugging_face_model(self):
        pass

class ChatBotQuizAI(ChatBotAI):
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

    def __init__(self, model=None, tokenizer=None):
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

        super().__init__(model=model, tokenizer=tokenizer)
        self.quiz_mode = False
        self.current_answer = None

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

        prompt = """
        ### Instruction:
        Generate ONE chemistry quiz question.

        ### Constraints:
        - Do NOT write code
        - Do NOT explain
        - Only output quiz

        ### Output Format:
        QUESTION: <question>
        ANSWER: <answer>

        ### Response:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.9,
                do_sample=True,
                top_p=0.95
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()
        print(f"quiz response: {response}")
        question, answer = None, None
        for line in generated.split("\n"):
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
        return "Type 'quiz' to start quiz mode."
        
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

        prompt = f"""
        You are a chemistry teacher grading a student's answer.

        Correct answer:
        {self.current_answer}

        Student answer:
        {student_answer}

        Respond ONLY:
        CORRECT or INCORRECT
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.2
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
