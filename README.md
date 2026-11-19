# researchpaper-qa-assistant
# Paper Q&A – Research Paper Assistant using RAG

An AI-powered research paper assistant that allows users to upload scientific research papers in PDF format, ask questions about their content, and generate summaries.

Built using **Retrieval-Augmented Generation (RAG)**, the application retrieves relevant sections from uploaded research papers and uses a Large Language Model (LLM) to generate context-aware answers with source references.

## 🚀 Live Demo

<https://huggingface.co/spaces/aditiattu/researchpaper-qa-assistant>

**Status:** Deployed successfully on Hugging Face Spaces.

## ✨ Features

- Upload scientific research papers in PDF format.
- Extract text using PyMuPDF.
- Divide extracted content into meaningful chunks.
- Generate semantic embeddings using `BAAI/bge-small-en-v1.5`.
- Store and search embeddings using FAISS.
- Ask questions and receive context-aware answers.
- View relevant retrieved source passages.
- Generate research paper summaries.

## 🧠 RAG Architecture

```text
Research Paper (PDF)
        |
        v
  Text Extraction
     (PyMuPDF)
        |
        v
    Text Chunking
        |
        v
 Embedding Generation
 (BGE-small-en-v1.5)
        |
        v
  FAISS Vector Index
        |
        v
   User Question
        |
        v
 Semantic Retrieval
        |
        v
 Relevant Text Chunks
        |
        v
  Groq LLM Inference
        |
        v
 Answer + Source Passages
```

## 🛠️ Tech Stack

- Python
- Gradio
- PyMuPDF
- Sentence Transformers
- BAAI/bge-small-en-v1.5
- FAISS
- Groq API
- Hugging Face Spaces

## 📁 Project Structure

```text
Paper-QA/
├── app.py
├── rag.py
├── requirements.txt
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/aditiattu13/researchpaper-qa-assistant.git
cd researchpaper-qa-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Set the following variables using your Groq API credentials:

```text
LLM_API_KEY=your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Never commit your API key to GitHub.

### 5. Start the application

```bash
python app.py
```

## ☁️ Deployment

Deployed using Hugging Face Spaces with Gradio.

**Live Application:**  
https://huggingface.co/spaces/aditiattu/researchpaper-qa-assistant

API credentials are configured through Hugging Face Space Secrets and Variables.

## 🎯 Project Objective

To simplify the process of exploring scientific literature by allowing users to interact with research papers through natural-language questions and AI-generated summaries.

The project demonstrates the integration of document processing, semantic search, vector databases, and Large Language Models through a Retrieval-Augmented Generation pipeline.

## 👩‍💻 Author

**Aditi Attupuram**

GitHub: [@aditiattu13](https://github.com/aditiattu13)

---

*An AI-powered research assistant built using Retrieval-Augmented Generation.*
