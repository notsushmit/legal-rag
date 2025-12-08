"""
Utility functions for Legal Assistant RAG Chatbot.

Provides PDF text extraction with OCR fallback, filename metadata parsing,
file listing utilities, and text cleaning/normalization functions.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pdfplumber
import pytesseract
from PIL import Image
import io


def extract_text_from_pdf(pdf_path: Path, use_ocr: bool = False) -> Tuple[str, List[Dict]]:
    """
    Extract text from a PDF file with optional OCR fallback.
    
    Args:
        pdf_path: Path to the PDF file
        use_ocr: If True, force OCR extraction
    
    Returns:
        Tuple of (full_text, page_metadata_list)
        page_metadata_list contains dicts with page_number, text, char_count
    """
    full_text = ""
    page_metadata = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = ""
                
                if not use_ocr:
                    # Try standard text extraction first
                    page_text = page.extract_text() or ""
                
                # If no text extracted or OCR forced, use OCR
                if not page_text.strip() or use_ocr:
                    try:
                        # Convert page to image and OCR
                        img = page.to_image(resolution=300)
                        pil_img = img.original
                        page_text = pytesseract.image_to_string(pil_img)
                    except Exception as ocr_error:
                        print(f"OCR failed for page {page_num} of {pdf_path}: {ocr_error}")
                        page_text = ""
                
                full_text += page_text + "\n\n"
                page_metadata.append({
                    "page_number": page_num,
                    "text": page_text,
                    "char_count": len(page_text)
                })
    
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return "", []
    
    return full_text, page_metadata


def parse_filename_metadata(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse metadata from filename using heuristics.
    
    Expected patterns:
    - Court abbreviations: SC, HC, DC
    - Years: 4-digit numbers
    - Case numbers: patterns like "2023_SC_1234"
    
    Args:
        filename: Name of the file (without path)
    
    Returns:
        Dict with parsed metadata (court, year, case_number, etc.)
    """
    metadata = {
        "court": None,
        "year": None,
        "case_number": None,
        "act_name": None
    }
    
    # Remove extension
    name = Path(filename).stem
    
    # Try to extract court
    court_match = re.search(r'\b(SC|HC|DC)\b', name, re.IGNORECASE)
    if court_match:
        metadata["court"] = court_match.group(1).upper()
    
    # Try to extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', name)
    if year_match:
        metadata["year"] = year_match.group(0)
    
    # Try to extract case number
    case_match = re.search(r'\d{4}_[A-Z]+_\d+', name)
    if case_match:
        metadata["case_number"] = case_match.group(0)
    
    # Check for common act names
    act_patterns = {
        "IPC": r'\bIPC\b',
        "CrPC": r'\bCrPC\b',
        "Evidence Act": r'\bEvidence\s*Act\b',
        "Constitution": r'\bConstitution\b'
    }
    
    for act_name, pattern in act_patterns.items():
        if re.search(pattern, name, re.IGNORECASE):
            metadata["act_name"] = act_name
            break
    
    return metadata


def parse_case_metadata_from_text(text: str) -> Dict[str, Optional[str]]:
    """
    Parse case metadata from judgment text using regex heuristics.
    
    Looks for:
    - Case name/title
    - Bench composition
    - Judgment date
    - Citation
    
    Args:
        text: First few pages of judgment text
    
    Returns:
        Dict with parsed metadata
    """
    metadata = {
        "case_name": None,
        "bench": None,
        "judgment_date": None,
        "citation": None
    }
    
    # Limit to first 2000 characters for performance
    header_text = text[:2000]
    
    # Try to extract case name (usually in ALL CAPS at the beginning)
    case_name_match = re.search(
        r'^([A-Z][A-Z\s&.,()]+)\s+(?:vs?\.?|versus)\s+([A-Z][A-Z\s&.,()]+)',
        header_text,
        re.MULTILINE
    )
    if case_name_match:
        metadata["case_name"] = f"{case_name_match.group(1).strip()} v. {case_name_match.group(2).strip()}"
    
    # Try to extract bench
    bench_match = re.search(r'BENCH:\s*(.+?)(?:\n|$)', header_text, re.IGNORECASE)
    if bench_match:
        metadata["bench"] = bench_match.group(1).strip()
    
    # Try to extract date
    date_match = re.search(
        r'(?:DATE|DECIDED ON|JUDGMENT DATE):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        header_text,
        re.IGNORECASE
    )
    if date_match:
        metadata["judgment_date"] = date_match.group(1)
    
    # Try to extract citation
    citation_match = re.search(r'\((\d{4})\)\s+(\d+)\s+([A-Z]+)\s+(\d+)', header_text)
    if citation_match:
        metadata["citation"] = f"({citation_match.group(1)}) {citation_match.group(2)} {citation_match.group(3)} {citation_match.group(4)}"
    
    return metadata


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    - Remove excessive whitespace
    - Normalize line breaks
    - Preserve section markers
    - Remove page numbers and headers/footers
    
    Args:
        text: Raw extracted text
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove common page number patterns
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
    
    # Remove excessive dashes or underscores (often used as separators)
    text = re.sub(r'[-_]{5,}', '', text)
    
    return text.strip()


def list_pdf_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    List all PDF files in a directory.
    
    Args:
        directory: Directory to search
        recursive: If True, search subdirectories
    
    Returns:
        List of PDF file paths
    """
    if recursive:
        return list(directory.rglob("*.pdf"))
    else:
        return list(directory.glob("*.pdf"))


def chunk_text_with_metadata(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    page_metadata: List[Dict]
) -> List[Dict]:
    """
    Chunk text while preserving page number metadata.
    
    Args:
        text: Full text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        page_metadata: List of page metadata dicts
    
    Returns:
        List of chunk dicts with text, chunk_index, and page_number
    """
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Try to find page number for this chunk
        # Simple heuristic: use the page that contains the start position
        char_count = 0
        page_number = 1
        for page in page_metadata:
            char_count += page["char_count"]
            if char_count >= start:
                page_number = page["page_number"]
                break
        
        chunks.append({
            "text": chunk_text,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "start_char": start,
            "end_char": end
        })
        
        chunk_index += 1
        start = end - chunk_overlap
    
    return chunks
