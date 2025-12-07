"""
FastAPI application for legal-rag-vibe.

Exposes three main endpoints:
- POST /research: Legal research assistant
- POST /judgment: Judgment simulation/reference
- POST /summarize: Headnote generation

Each endpoint assembles prompts, calls retriever + LLM, verifies citations,
logs requests, and returns JSON responses with provenance and disclaimers.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn

from src.config import Config
from src.retriever import get_retriever
from src.llm_client import get_llm_client
from src.prompt_templates import (
    build_research_prompt,
    build_judgment_prompt,
    build_summarize_prompt
)
from src.verify_and_log import (
    verify_bracket_citations,
    create_log_entry,
    write_log_file,
    should_retry_generation,
    build_retry_prompt
)

# Initialize FastAPI app
app = FastAPI(
    title="legal-rag-vibe",
    description="Legal RAG chatbot for Indian law",
    version="1.0.0"
)


# Request models
class ResearchRequest(BaseModel):
    q: str
    task: Optional[str] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


class JudgmentRequest(BaseModel):
    facts: str
    mode: str  # "hypothetical" or "reference"
    top_k: Optional[int] = None
    temperature: Optional[float] = None


class SummarizeRequest(BaseModel):
    query: Optional[str] = None
    case_text: Optional[str] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


# Response models
class APIResponse(BaseModel):
    mode: str
    answer: str
    retrieved: Optional[List[Dict]] = None
    verification: Optional[Dict] = None
    logfile: str
    disclaimer: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "legal-rag-vibe",
        "version": "1.0.0",
        "endpoints": ["/research", "/judgment", "/summarize"],
        "status": "running"
    }


@app.post("/research", response_model=APIResponse)
async def research(request: ResearchRequest):
    """Legal research assistant endpoint."""
    try:
        # Set defaults
        top_k = request.top_k or Config.DEFAULT_TOP_K
        temperature = request.temperature if request.temperature is not None else Config.RESEARCH_TEMPERATURE
        
        # Retrieve documents
        retriever = get_retriever()
        retrieved = retriever.retrieve(request.q, top_k=top_k)
        
        if not retrieved:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Build prompt
        prompt = build_research_prompt(request.q, retrieved)
        
        # Generate response
        llm_client = get_llm_client()
        result = llm_client.generate(prompt, temperature=temperature)
        answer = result["text"]
        
        # Verify citations
        verification = verify_bracket_citations(answer, len(retrieved))
        
        # Retry if invalid citations found
        if should_retry_generation(verification):
            retry_prompt = build_retry_prompt(prompt, len(retrieved), verification["invalid"])
            result = llm_client.generate(retry_prompt, temperature=temperature)
            answer = result["text"]
            verification = verify_bracket_citations(answer, len(retrieved))
        
        # Log request
        log_entry = create_log_entry(
            mode="research",
            user_input=request.q,
            retrieved=retrieved,
            prompt=prompt,
            llm_response=answer,
            verification=verification,
            temperature=temperature
        )
        logfile = write_log_file(log_entry, "research")
        
        return APIResponse(
            mode="research",
            answer=answer,
            retrieved=[{"metadata": doc["metadata"]} for doc in retrieved],
            verification=verification,
            logfile=str(logfile),
            disclaimer="For research/educational use only."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/judgment", response_model=APIResponse)
async def judgment(request: JudgmentRequest):
    """Judgment simulation/reference endpoint."""
    try:
        # Validate mode
        if request.mode not in ["hypothetical", "reference"]:
            raise HTTPException(status_code=400, detail="Mode must be 'hypothetical' or 'reference'")
        
        # Set defaults
        top_k = request.top_k or Config.DEFAULT_TOP_K
        temperature = request.temperature if request.temperature is not None else Config.JUDGMENT_TEMPERATURE
        
        # Retrieve documents
        retriever = get_retriever()
        retrieved = retriever.retrieve(request.facts, top_k=top_k)
        
        if not retrieved:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Build prompt
        prompt = build_judgment_prompt(request.facts, request.mode, retrieved)
        
        # Generate response
        llm_client = get_llm_client()
        result = llm_client.generate(prompt, temperature=temperature)
        answer = result["text"]
        
        # Verify citations
        verification = verify_bracket_citations(answer, len(retrieved))
        
        # Retry if invalid citations found
        if should_retry_generation(verification):
            retry_prompt = build_retry_prompt(prompt, len(retrieved), verification["invalid"])
            result = llm_client.generate(retry_prompt, temperature=temperature)
            answer = result["text"]
            verification = verify_bracket_citations(answer, len(retrieved))
        
        # Log request
        log_entry = create_log_entry(
            mode="judgment",
            user_input=request.facts,
            retrieved=retrieved,
            prompt=prompt,
            llm_response=answer,
            verification=verification,
            temperature=temperature
        )
        logfile = write_log_file(log_entry, "judgment")
        
        disclaimer = "HYPOTHETICAL ANALYSIS — NOT LEGAL ADVICE" if request.mode == "hypothetical" else "REFERENCE ANALYSIS — NOT LEGAL ADVICE"
        
        return APIResponse(
            mode="judgment",
            answer=answer,
            retrieved=[{"metadata": doc["metadata"]} for doc in retrieved],
            verification=verification,
            logfile=str(logfile),
            disclaimer=disclaimer
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=APIResponse)
async def summarize(request: SummarizeRequest):
    """Summarization/headnote generation endpoint."""
    try:
        # Set defaults
        top_k = request.top_k or 3
        temperature = request.temperature if request.temperature is not None else Config.SUMMARIZE_TEMPERATURE
        
        retrieved = []
        
        # Either use case_text or retrieve by query
        if not request.case_text and not request.query:
            raise HTTPException(status_code=400, detail="Either 'query' or 'case_text' must be provided")
        
        if request.query and not request.case_text:
            retriever = get_retriever()
            retrieved = retriever.retrieve(request.query, top_k=top_k)
            
            if not retrieved:
                raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Build prompt
        prompt = build_summarize_prompt(request.query or "", retrieved, request.case_text)
        
        # Generate response
        llm_client = get_llm_client()
        result = llm_client.generate(prompt, temperature=temperature)
        answer = result["text"]
        
        # Log request
        log_entry = create_log_entry(
            mode="summarize",
            user_input=request.query or request.case_text[:500],
            retrieved=retrieved,
            prompt=prompt,
            llm_response=answer,
            verification={},
            temperature=temperature
        )
        logfile = write_log_file(log_entry, "summarize")
        
        return APIResponse(
            mode="summarize",
            answer=answer,
            retrieved=[{"metadata": doc["metadata"]} for doc in retrieved] if retrieved else None,
            logfile=str(logfile)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
