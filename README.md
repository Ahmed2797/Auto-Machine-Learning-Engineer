# Auto-Machine-Learning-Engineer

Developed a comprehensive Auto-Machine-Learning (Auto-ML) Pipeline designed to democratize data science by automating the end-to-end model development lifecycle. The system handles everything from automated feature engineering to hyperparameter optimization, reducing the time-to-deployment from weeks to hours.

## 🤖 AI-Agent Repository Replicator (CrewAI + FAISS)

An advanced, self-healing automated software engineering pipeline that replicates existing codebases using Retrieval-Augmented Generation (RAG). This system doesn't just copy files—it understands the original logic via a FAISS vector index and reconstructs a high-quality, industry-standard version of the project.

## 🌟 Key Features

* **RAG-Driven Code Generation:** Uses a FAISS index of your original repository to maintain coding styles, import patterns, and logic.
* **Self-Healing Loop:** Integrated `execute_and_debug` tool allows the agent to run generated code, capture tracebacks, and fix errors automatically.
* **Safe Sandboxing:** All operations are restricted to a `replicated_project_output` directory to protect your local environment.
* **Industry Standards:** Automatically implements Python PEP8 standards, structured logging, and robust error handling.
* **Dynamic Task Orchestration:** Powered by **CrewAI**, managing dependencies and file-by-file implementation sequentially.

---

## 🏗️ Architecture

The system operates in three distinct phases:

1.**Discovery:** The Architect agent scans the FAISS index to map out the original folder and file structure.
2.**Implementation:** The Developer agent uses custom tools to generate code for each empty file based on retrieved context.
3.**Validation:** Every file is executed in a subprocess; if a `SyntaxError` or `ModuleNotFoundError` is detected, the agent self-corrects until it passes.

---

## 🛠️ Tech Stack

* **Orchestration:** [CrewAI](https://github.com/joaomoura/crewai)
* **LLMs:** GPT-4o / GPT-4o-mini (OpenAI)
* **Vector Database:** FAISS
* **Embeddings:** OpenAI `text-embedding-3-small`
* **Environment:** Python 3.10+

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have an OpenAI API Key and your source repository indexed in FAISS.

```bash
export OPENAI_API_KEY='your_api_key_here'
```

### Project Setup

#### Clone this orchestrator

```bash
git clone [https://github.com/Ahmed2797/Auto-Machine-Learning-Engineer](https://github.com/Ahmed2797/Auto-Machine-Learning-Engineer)

### Install dependencies
pip install crewai langchain-openai faiss-cpu

## store database or embedding
python store.py
## demo 
python main.py
## run crew
python app.py
