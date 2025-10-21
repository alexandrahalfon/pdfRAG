from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import tempfile
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

# Import your pipeline modules
from doc_processing import process_pdf_complete
from chunking_step import semantic_chunk_document
from embedding_step import generate_embeddings, save_embeddings
from similarity_search import load_embeddings_from_files
from generation_step import ask_document

load_dotenv()

app = FastAPI(
    title="Document RAG API",
    description="Upload PDFs and ask questions using RAG",
    version="1.0.0"
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]

class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks: int

# Global state
class DocumentStore:
    def __init__(self):
        self.client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
        self.embeddings = None
        self.metadata = None
        self.is_ready = False
        self._load_embeddings()
    
    def _load_embeddings(self):
        try:
            self.embeddings, self.metadata = load_embeddings_from_files()
            self.is_ready = True
        except:
            self.is_ready = False

store = DocumentStore()

@app.get("/")
async def root():
    return {"message": "Document RAG API", "endpoints": ["/upload", "/query", "/status", "/clear"]}

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents"""
    if not files or not all(f.filename.lower().endswith('.pdf') for f in files):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    all_chunks = []
    processed_files = []
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_path = tmp_file.name
        
        try:
            structured_json_path = process_pdf_complete(temp_path, store.client)
            chunks = semantic_chunk_document(structured_json_path, save_to_disk=False)
            for chunk in chunks:
                chunk['source_file'] = file.filename
            all_chunks.extend(chunks)
            processed_files.append(file.filename)
        finally:
            os.unlink(temp_path)
    
    if all_chunks:
        chunks_with_embeddings = generate_embeddings(all_chunks, store.client, save_npy=False)
        save_embeddings(chunks_with_embeddings)
        store._load_embeddings()
    
    return UploadResponse(
        message="Documents processed successfully",
        files_processed=processed_files,
        total_chunks=len(all_chunks)
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask questions about uploaded documents"""
    if not store.is_ready:
        raise HTTPException(status_code=400, detail="No documents available. Please upload PDF files first.")
    
    result = ask_document(request.question, store.embeddings, store.metadata, store.client, request.top_k, False)
    return QueryResponse(answer=result['answer'], sources=result['sources'])

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "ready" if store.is_ready else "not_ready",
        "total_chunks": len(store.metadata) if store.metadata else 0,
        "documents_count": len(set(chunk.get('source_file', 'unknown') for chunk in store.metadata)) if store.metadata else 0
    }

@app.delete("/clear")
async def clear_documents():
    """Clear all documents and embeddings"""
    embeddings_dir = Path("./embeddings")
    if embeddings_dir.exists():
        shutil.rmtree(embeddings_dir)
    store.embeddings = None
    store.metadata = None
    store.is_ready = False
    return {"message": "All documents cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)