# DocuRAG-FastAPI 🚀

A production-ready Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **LangChain**, and **ChromaDB**.

This project demonstrates how to move beyond basic LangChain scripts to a scalable REST API architecture suitable for AI agents and enterprise applications. It includes document chunking, embeddings, vector storage, and an OpenAI-powered query endpoint.

## Features
- **FastAPI Backend:** Fully asynchronous, typed, and auto-documented (Swagger UI).
- **Document Processing:** Handles PDF and TXT uploads with automated text splitting.
- **Local Vector Store:** Uses ChromaDB for fast, local embedding retrieval.
- **LangChain Integration:** Utilizes LangChain's latest `create_retrieval_chain` architecture.
- **Source Tracking:** Returns exact document sources used to formulate answers to prevent hallucinations.

## Tech Stack
- **Python 3.10+**
- **FastAPI** & Uvicorn
- **LangChain** (Core, Community, OpenAI)
- **ChromaDB**
- **OpenAI API** (`gpt-3.5-turbo` & `text-embedding-3-small`)

## Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/DocuRAG-FastAPI.git
cd DocuRAG-FastAPI
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Copy the example env file and add your OpenAI key:
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

### 3. Run the Server
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.
View the Swagger documentation at `http://localhost:8000/docs`.

## API Endpoints

### `POST /api/upload`
Uploads a document, chunks it, embeds it, and stores it in ChromaDB.
- **Body:** `multipart/form-data` containing `file` (.pdf or .txt)

### `POST /api/query`
Queries the vector database and generates an answer using an LLM.
- **Body:** `{"query": "What is the document about?"}`
- **Response:**
```json
{
  "answer": "The document describes...",
  "sources": ["./uploaded_docs/example.pdf"]
}
```

## Why this architecture?
Many AI projects fail when moving to production because they are built as monolithic scripts. By wrapping the RAG pipeline in FastAPI:
- We can decouple the AI logic from the frontend (React/Next.js/SwiftUI).
- We gain built-in data validation via Pydantic.
- We can easily scale horizontally in a Dockerized environment (e.g., AWS ECS or Kubernetes).

---
*Built by Rajnish Singh.*
