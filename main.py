import os
import aiofiles
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration and Settings Validation
class Settings(BaseSettings):
    openai_api_key: str
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

UPLOAD_DIR = "./uploaded_docs"
CHROMA_DB_DIR = "./chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Lifespan Context Manager (replaces global state initialization)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Initializing Vector Store and Models...")
    
    try:
        # Initialize Embeddings & LLM
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=settings.openai_api_key)
        
        # Initialize ChromaDB connection once
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        
        # Attach to application state
        app.state.vectorstore = vectorstore
        app.state.llm = llm
        app.state.embeddings = embeddings
        
        print("Initialization complete.")
    except Exception as e:
        print(f"Failed to initialize core components: {e}")
        raise e
        
    yield
    
    print("Shutting down...")

app = FastAPI(
    title="DocuRAG API",
    description="Production-ready FastAPI backend for Retrieval-Augmented Generation.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencies
def get_vectorstore(request: Request) -> Chroma:
    return request.app.state.vectorstore

def get_llm(request: Request) -> ChatOpenAI:
    return request.app.state.llm

def get_embeddings(request: Request) -> OpenAIEmbeddings:
    return request.app.state.embeddings

class QueryRequest(BaseModel):
    query: str

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    vectorstore: Chroma = Depends(get_vectorstore),
    embeddings: OpenAIEmbeddings = Depends(get_embeddings)
):
    """
    Asynchronously upload a document, chunk it, and index it in the vector database.
    """
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Asynchronous File I/O
    async with aiofiles.open(file_path, 'wb') as out_file:
        while content := await file.read(1024 * 1024):  # 1MB chunks
            await out_file.write(content)
            
    try:
        # 1. Load Document
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        docs = loader.load()
        
        # 2. Chunk Document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Store in Vector DB using the injected embedding function
        vectorstore.add_documents(documents=splits)
        
        return {"message": f"Successfully processed and indexed {file.filename}", "chunks": len(splits)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_documents(
    request: QueryRequest,
    vectorstore: Chroma = Depends(get_vectorstore),
    llm: ChatOpenAI = Depends(get_llm)
):
    """
    Query the indexed documents using RAG with MMR (Maximal Marginal Relevance) search.
    """
    try:
        # Define Retriever with MMR for diversity
        retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 4, "fetch_k": 20}
        )
        
        # Define RAG Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create Chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Execute Query
        response = await rag_chain.ainvoke({"input": request.query})
        
        # Extract sources
        sources = [doc.metadata.get('source', 'Unknown') for doc in response['context']]
        
        return {
            "answer": response['answer'],
            "sources": list(set(sources))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
