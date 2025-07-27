# 🍕 Restaurant AI Agent

An intelligent AI-powered assistant tailored for a specific restaurant. This agent leverages restaurant review data and modern generative AI technologies to answer customer queries in natural language.

---

## 🚀 Features

- 💬 Chat interface for answering customer questions
- 🧠 Retrieval-Augmented Generation (RAG) with fine-tuned data
- 🔍 Query-specific document retrieval using ChromaDB
- 🦙 Embedded vector storage with Ollama (`mxbai-embed-large`)
- 🤖 Local LLM response generation with Ollama (`llama3.2`)
- 📦 Auto-logs all Q&A interactions into a searchable collection

---

## 🧰 Tech Stack

| Tool          | Purpose                                  |
|---------------|------------------------------------------|
| **LangChain** | Agent orchestration and chaining         |
| **ChromaDB**  | Vector storage and retrieval             |
| **Ollama**    | Local LLM and embedding generation       |
| **RAG**       | Injects retrieved reviews into prompts   |
| **Python**    | Core programming environment             |

---

## 📁 Project Prerequisites
- Python 3 or above
- Ollama with llama3.2 in your machine
- ChromaDB Cloud setup

## Getting Started

Clone the project

```bash
git clone https://github.com/shubham-singh0109/Agentic_AI.git
```

Go to the project directory

```bash
cd Agentic_AI
```

## Setting up Ollama in the local machine

Download https://ollama.com

### In your Terminal

```bash
ollama
```
```bash
ollama pull llama3.2
```
```bash
ollama pull mxbai-embed-large
```
```bash
ollama list
```

## Create .env file to setup your tokens and API key from ChromaDB

```bash
ChromaDB_Token_Key=<YOUR_TOKEN>
ChromaDB_API_Key=<YOUR_API_KEY>
```

## Create Virtual Environment

```bash
python -m venv venv
```
```bash
source ./venv/bin/activate
```

## Install dependencies and libraries

```bash
pip install -r requirement.txt
```

## Run Locally

```bash
python main.py
# or
python3 main.py
```