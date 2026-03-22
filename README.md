# ChemBot – Chemistry Chatbot & Quiz AI

**Author:** Md Ismail Quraishi  
**Date:** 14/03/2026  

---

## Overview

ChemBot is a **chemistry-focused AI chatbot** designed to answer chemistry questions and provide interactive quizzes. It leverages **LangChain** and **Ollama** to run a local language model for generating accurate chemistry responses and grading quiz answers.  

The application provides two main modes:

1. **Normal Chat Mode** – Ask chemistry questions and get explanations.  
2. **Quiz Mode** – Interactive quiz for testing chemistry knowledge.

---

## Features

- Answer chemistry-related questions only.  
- Interactive quiz mode with auto-grading.  
- Maintains short conversation history for context.  
- Runs entirely locally via Docker.  

---

## Project Structure

```
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── app.py
    └── chembot_ai.py
```


### File Descriptions

- **`src/app.py`** – Main entry point. Handles user input and toggles between normal chat and quiz mode.  
- **`src/chembot_ai.py`** – Core AI logic: chemistry question answering, quiz generation, and grading.  
- **`Dockerfile`** – Containerizes the project for easy deployment.  
- **`requirements.txt`** – Python dependencies.  
- **`LICENSE`** – Project license.  

---

## Requirements

- Docker  
- Python 3.11+ (for local testing outside Docker)  
- Ollama host and model configured via `.env` file:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=chem_bot_model_name
```

## Installation & Setup

### 1. Build the Docker image

```bash
docker build -t chembot .
```

### 2. Run the container

```bash
docker run -it   --network="host"   --env-file .env   -v ~/.cache/huggingface:/root/.cache/huggingface   -v $(pwd)/faiss_index:/app/faiss_index   chembot
```

## Normal Chat Mode

```
You: What is the periodic table?
ChemBot: The periodic table is a tabular arrangement of chemical elements...
```

## Quiz Mode

```
You: quiz
ChemBot: Quiz Mode Started!

QUESTION: What is the atomic number of oxygen?

You: 8
ChemBot: Correct! 🎉
```