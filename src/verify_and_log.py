"""
Citation verification and logging module for legal-rag-vibe.

Provides citation verification to detect hallucinated references,
logging/audit file writing, and retry logic for failed verifications.
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from src.config import Config


def verify_bracket_citations(text: str, num_retrieved: int) -> Dict[str, List[int]]:
    """
    Verify that all bracket citations in text are valid.
    
    Args:
        text: Generated text to verify
        num_retrieved: Number of retrieved passages (valid range: 1 to num_retrieved)
    
    Returns:
        Dict with 'valid' and 'invalid' lists of citation numbers
    """
    # Find all bracket citations [n]
    citations = re.findall(r'\[(\d+)\]', text)
    
    valid = []
    invalid = []
    
    for citation in citations:
        num = int(citation)
        if 1 <= num <= num_retrieved:
            if num not in valid:
                valid.append(num)
        else:
            if num not in invalid:
                invalid.append(num)
    
    return {
        "valid": sorted(valid),
        "invalid": sorted(invalid)
    }


def detect_unverified_citations(text: str, retrieved_metadata: List[Dict]) -> List[str]:
    """
    Detect case names or citations in text that aren't bracketed.
    
    Args:
        text: Generated text
        retrieved_metadata: Metadata from retrieved documents
    
    Returns:
        List of potentially unverified citations
    """
    unverified = []
    
    # Pattern for case citations like "(2023) 5 SCC 123"
    citation_pattern = r'\(\d{4}\)\s+\d+\s+[A-Z]+\s+\d+'
    found_citations = re.findall(citation_pattern, text)
    
    # Check if these citations match retrieved metadata
    for citation in found_citations:
        matched = False
        for meta in retrieved_metadata:
            if meta.get("citation") == citation:
                matched = True
                break
        
        if not matched:
            unverified.append(citation)
    
    return unverified


def create_log_entry(
    mode: str,
    user_input: str,
    retrieved: List[Dict],
    prompt: str,
    llm_response: str,
    verification: Dict,
    temperature: float,
    user_id: Optional[str] = None
) -> Dict:
    """
    Create a structured log entry for a request.
    
    Args:
        mode: Request mode (research/judgment/summarize)
        user_input: User's query or facts
        retrieved: Retrieved documents with metadata
        prompt: Assembled prompt sent to LLM
        llm_response: Raw LLM response
        verification: Citation verification results
        temperature: Generation temperature used
        user_id: Optional user identifier
    
    Returns:
        Dict with complete log entry
    """
    # Truncate long fields
    def truncate(text: str, max_len: int = 500) -> str:
        return text[:max_len] + "..." if len(text) > max_len else text
    
    # Extract metadata summaries
    retrieved_summary = []
    for doc in retrieved:
        meta = doc.get("metadata", {})
        retrieved_summary.append({
            "id": doc.get("id"),
            "source_file": meta.get("source_file"),
            "page_number": meta.get("page_number"),
            "chunk_index": meta.get("chunk_index"),
            "case_name": meta.get("case_name"),
            "distance": doc.get("distance")
        })
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "mode": mode,
        "user_input": truncate(user_input, 1000),
        "retrieved_count": len(retrieved),
        "retrieved_metadata": retrieved_summary,
        "prompt": truncate(prompt, 2000),
        "temperature": temperature,
        "llm_response": truncate(llm_response, 2000),
        "verification": verification,
        "full_response_length": len(llm_response)
    }
    
    return log_entry


def write_log_file(log_entry: Dict, mode: str) -> Path:
    """
    Write log entry to a JSON file.
    
    Args:
        log_entry: Log entry dict
        mode: Request mode (for filename)
    
    Returns:
        Path to the created log file
    """
    Config.ensure_directories()
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{mode}_{timestamp}.json"
    filepath = Config.LOGS_DIR / filename
    
    # Write log file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    return filepath


def should_retry_generation(verification: Dict) -> bool:
    """
    Determine if generation should be retried based on verification.
    
    Args:
        verification: Citation verification results
    
    Returns:
        True if retry is recommended
    """
    # Retry if there are invalid citations
    return len(verification.get("invalid", [])) > 0


def build_retry_prompt(original_prompt: str, num_retrieved: int, invalid_citations: List[int]) -> str:
    """
    Build a stricter prompt for retry after citation verification failure.
    
    Args:
        original_prompt: Original prompt
        num_retrieved: Number of retrieved passages
        invalid_citations: List of invalid citation numbers found
    
    Returns:
        Modified prompt with stricter instructions
    """
    retry_instruction = f"""
CRITICAL CORRECTION REQUIRED:
Your previous response contained invalid citations: {invalid_citations}
You MUST use ONLY bracket numbers from [1] to [{num_retrieved}].
Do NOT use any other numbers in bracket citations.

"""
    
    # Insert retry instruction at the beginning
    return retry_instruction + original_prompt
