# DocuRAG-FastAPI 🚀

A production-ready Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **LangChain**, and **ChromaDB**.

This project demonstrates how to move beyond basic LangChain scripts to a highly scalable, async REST API architecture suitable for enterprise applications.

## Production-Grade Features (v2.0)
- **Advanced Application Lifecycles**: Initializes the Vector Store (Chroma) and LLMs via FastAPI's `@asynccontextmanager lifespan`, ensuring they are loaded into memory exactly once at server startup (prevents connection thrashing).
- **Dependency Injection**: Utilizes FastAPI's `Depends()` to inject the global database and LLM connections into route handlers safely and cleanly.
- **Strict Configuration**: Uses `pydantic-settings` to strictly enforce environment variables at startup, preventing runtime crashes.
- **Asynchronous File I/O**: Upload endpoints use `aiofiles` to save document streams asynchronously without blocking the main event loop, allowing for high concurrent throughput.
- **Anti-Hallucination via MMR**: Upgraded the retriever to use Maximal Marginal Relevance (`search_type="mmr"`) to ensure the LLM receives diverse, highly relevant chunks, while strictly citing the exact source documents in the response.

## Tech Stack
- **Python 3.10+**
- **FastAPI** & Uvicorn
- **LangChain**
- **ChromaDB**
- **OpenAI API** (`gpt-3.5-turbo` & `text-embedding-3-small`)
- **Pydantic v2**

## Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/givemehat/DocuRAG-FastAPI.git
cd DocuRAG-FastAPI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
Copy the example env file and add your OpenAI key:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your_key_here
```

### 3. Run the Server
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.
View the Swagger documentation at `http://localhost:8000/docs`.

---
*Architected by Rajnish Singh.*
