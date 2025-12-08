"""
Prompt templates for Legal Assistant RAG Chatbot.

Contains textual prompt templates for research, judgment simulation,
and summarization modes. Templates include system instructions,
retrieved passage formatting, and output format specifications.
"""

from typing import List, Dict


def format_retrieved_passages(retrieved: List[Dict]) -> str:
    """
    Format retrieved passages with bracket numbers and metadata.
    
    Args:
        retrieved: List of retrieved document dicts
    
    Returns:
        Formatted string with numbered passages
    """
    passages = []
    for i, doc in enumerate(retrieved, start=1):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown")
        page = metadata.get("page_number", "?")
        case_name = metadata.get("case_name", "")
        
        header = f"[{i}] Source: {source}, Page: {page}"
        if case_name:
            header += f", Case: {case_name}"
        
        passage = f"{header}\n{doc['document']}\n"
        passages.append(passage)
    
    return "\n".join(passages)


def build_research_prompt(query: str, retrieved: List[Dict]) -> str:
    """
    Build prompt for legal research mode.
    
    Args:
        query: User's research question
        retrieved: Retrieved documents
    
    Returns:
        Complete prompt string
    """
    passages = format_retrieved_passages(retrieved)
    
    prompt = f"""You are a legal research assistant for Indian law. Your task is to provide accurate, well-researched answers based ONLY on the provided legal documents.

RETRIEVED LEGAL PASSAGES:
{passages}

USER QUERY: {query}

INSTRUCTIONS:
1. Provide an executive summary (2-4 sentences) answering the query
2. List key points as bullet notes
3. If multiple cases or sections are relevant, compare them briefly
4. Cite sources using ONLY bracket numbers [1], [2], etc. that appear in the passages above
5. If the provided passages are insufficient to answer the query, clearly state so
6. End with a numbered sources list matching your bracket citations

OUTPUT FORMAT:
## Executive Summary
[Your 2-4 sentence summary with citations]

## Key Points
- [Point 1 with citation]
- [Point 2 with citation]
...

## Sources
1. [Source details from passage [1]]
2. [Source details from passage [2]]
...

CRITICAL: Use ONLY the bracket numbers [1] through [{len(retrieved)}] that correspond to the passages above. Do not invent citations or reference sources not provided.
"""
    return prompt


def build_judgment_prompt(facts: str, mode: str, retrieved: List[Dict]) -> str:
    """
    Build prompt for judgment simulation mode.
    
    Args:
        facts: Case facts provided by user
        mode: "hypothetical" or "reference"
        retrieved: Retrieved documents
    
    Returns:
        Complete prompt string with appropriate header
    """
    passages = format_retrieved_passages(retrieved)
    
    header_text = "HYPOTHETICAL ANALYSIS — NOT LEGAL ADVICE" if mode == "hypothetical" else "REFERENCE ANALYSIS — NOT LEGAL ADVICE"
    
    prompt = f"""You are simulating judicial reasoning for educational purposes. You must begin your response with the header: "{header_text}"

RETRIEVED LEGAL PASSAGES:
{passages}

CASE FACTS:
{facts}

INSTRUCTIONS:
1. Begin with the exact header: "{header_text}"
2. Analyze the facts in light of the retrieved legal passages
3. Structure your analysis as follows:
   - Facts: Restate the key facts
   - Issues: Identify legal issues raised
   - Reasoning: Simulate judicial reasoning using cautious language (may, could, likely, appears)
   - Hypothetical Holding(s): State potential conclusions
   - Sources: List all cited sources by bracket number

4. Use ONLY bracket citations [1], [2], etc. corresponding to the passages above
5. Use cautious, conditional language throughout
6. This is for educational purposes only - not actual legal advice

OUTPUT FORMAT:
{header_text}

## Facts
[Restate key facts]

## Issues
1. [Issue 1]
2. [Issue 2]
...

## Reasoning
[Detailed analysis with citations using cautious language]

## Hypothetical Holding(s)
[Potential conclusions with citations]

## Sources
1. [Source from [1]]
2. [Source from [2]]
...

CRITICAL: Cite ONLY using bracket numbers [1] through [{len(retrieved)}]. Do not invent case names or citations not present in the passages.
"""
    return prompt


def build_summarize_prompt(query: str, retrieved: List[Dict], case_text: str = None) -> str:
    """
    Build prompt for summarization/headnote generation mode.
    
    Args:
        query: Optional query for retrieval-based summarization
        retrieved: Retrieved documents (if query-based)
        case_text: Direct case text to summarize (if provided)
    
    Returns:
        Complete prompt string
    """
    if case_text:
        content = f"CASE TEXT TO SUMMARIZE:\n{case_text}"
    else:
        content = f"RETRIEVED PASSAGES:\n{format_retrieved_passages(retrieved)}"
    
    prompt = f"""You are a legal headnote generator for Indian case law. Generate a comprehensive headnote with study notes.

{content}

INSTRUCTIONS:
1. Extract and organize the following information:
   - Facts: Key factual background
   - Issue: Main legal question(s)
   - Holding: Court's decision/ruling
   - Ratio Decidendi: Legal principle established
   
2. Generate 5 study notes highlighting important aspects

3. If using retrieved passages, cite using bracket numbers [1], [2], etc.

OUTPUT FORMAT:
## Facts
[Concise factual background]

## Issue
[Main legal question]

## Holding
[Court's decision]

## Ratio Decidendi
[Legal principle established]

## Study Notes
1. [Important point 1]
2. [Important point 2]
3. [Important point 3]
4. [Important point 4]
5. [Important point 5]

CRITICAL: Base your summary ONLY on the provided text. Do not add external information.
"""
    return prompt
