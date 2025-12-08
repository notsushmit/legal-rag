"""
Ingestion pipeline for Legal Assistant RAG Chatbot.

Orchestrates the complete ingestion process:
1. Discovery of PDF files in data directories
2. Text extraction with OCR fallback
3. Metadata parsing from filenames and content
4. Text cleaning and normalization
5. Chunking with overlap and page metadata
6. Embedding generation
7. Upsert to ChromaDB with full metadata
"""

import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

from src.config import Config
from src.utils import (
    extract_text_from_pdf,
    parse_filename_metadata,
    parse_case_metadata_from_text,
    clean_text,
    list_pdf_files,
    chunk_text_with_metadata
)
from src.embeddings_ import get_embedding_model


class LegalDocumentIngestor:
    """Handles ingestion of legal documents into ChromaDB."""
    
    def __init__(self):
        """Initialize the ingestor with ChromaDB client and embedding model."""
        Config.ensure_directories()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION_NAME,
            metadata={"description": "Legal judgments and statutes for RAG"}
        )
        
        # Initialize embedding model
        self.embedding_model = get_embedding_model()
        
        print(f"Initialized ingestor with collection: {Config.CHROMA_COLLECTION_NAME}")
        print(f"Current document count: {self.collection.count()}")
    
    def load_manifest(self) -> Dict:
        """Load the download manifest."""
        manifest_path = Config.RAW_DATA_DIR / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"downloads": [], "metadata": {}}
    
    def ingest_pdf(self, pdf_path: Path, source_type: str = "judgment") -> int:
        """
        Ingest a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            source_type: Type of document ("judgment", "act", etc.)
        
        Returns:
            Number of chunks ingested
        """
        print(f"\nProcessing: {pdf_path.name}")
        
        # Extract text
        full_text, page_metadata = extract_text_from_pdf(pdf_path)
        
        if not full_text.strip():
            print(f"  WARNING: No text extracted from {pdf_path.name}")
            return 0
        
        # Parse metadata
        filename_meta = parse_filename_metadata(pdf_path.name)
        content_meta = parse_case_metadata_from_text(full_text)
        
        # Clean text
        cleaned_text = clean_text(full_text)
        
        # Chunk text
        chunks = chunk_text_with_metadata(
            cleaned_text,
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP,
            page_metadata
        )
        
        print(f"  Extracted {len(page_metadata)} pages, created {len(chunks)} chunks")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Create unique ID
            chunk_id = f"{pdf_path.stem}_{chunk['chunk_index']}"
            
            # Combine metadata
            metadata = {
                "source_file": pdf_path.name,
                "source_type": source_type,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "case_name": content_meta.get("case_name"),
                "court": filename_meta.get("court") or content_meta.get("bench"),
                "judgment_date": content_meta.get("judgment_date"),
                "citation": content_meta.get("citation"),
                "year": filename_meta.get("year"),
                "act_name": filename_meta.get("act_name"),
                "url": None  # Can be populated from manifest
            }
            
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            documents.append(chunk["text"])
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        # Generate embeddings
        print(f"  Generating embeddings...")
        embeddings = self.embedding_model.embed_texts(documents, batch_size=32)
        
        # Upsert to ChromaDB
        print(f"  Upserting to ChromaDB...")
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        
        print(f"  ✓ Ingested {len(chunks)} chunks from {pdf_path.name}")
        return len(chunks)
    
    def ingest_directory(self, directory: Path, source_type: str = "judgment") -> Dict[str, int]:
        """
        Ingest all PDFs from a directory.
        
        Args:
            directory: Directory containing PDFs
            source_type: Type of documents in this directory
        
        Returns:
            Dict with statistics
        """
        pdf_files = list_pdf_files(directory, recursive=True)
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return {"files": 0, "chunks": 0}
        
        print(f"\nFound {len(pdf_files)} PDF files in {directory}")
        
        total_chunks = 0
        successful_files = 0
        
        for pdf_path in tqdm(pdf_files, desc=f"Ingesting {source_type}s"):
            try:
                chunks = self.ingest_pdf(pdf_path, source_type)
                total_chunks += chunks
                if chunks > 0:
                    successful_files += 1
            except Exception as e:
                print(f"  ERROR processing {pdf_path.name}: {e}")
        
        return {
            "files": successful_files,
            "total_files": len(pdf_files),
            "chunks": total_chunks
        }
    
    def ingest_all(self) -> Dict[str, Dict[str, int]]:
        """
        Ingest all documents from all data directories.
        
        Returns:
            Dict with statistics for each directory
        """
        stats = {}
        
        # Ingest acts
        if Config.ACTS_DIR.exists():
            print("\n" + "="*60)
            print("INGESTING ACTS")
            print("="*60)
            stats["acts"] = self.ingest_directory(Config.ACTS_DIR, "act")
        
        # Ingest judgments
        if Config.JUDGMENTS_DIR.exists():
            print("\n" + "="*60)
            print("INGESTING JUDGMENTS")
            print("="*60)
            stats["judgments"] = self.ingest_directory(Config.JUDGMENTS_DIR, "judgment")
        
        # Ingest raw data
        if Config.RAW_DATA_DIR.exists():
            print("\n" + "="*60)
            print("INGESTING RAW DATA")
            print("="*60)
            stats["raw"] = self.ingest_directory(Config.RAW_DATA_DIR, "raw")
        
        # Print summary
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        for source, source_stats in stats.items():
            print(f"{source.upper()}: {source_stats['files']}/{source_stats.get('total_files', 0)} files, {source_stats['chunks']} chunks")
        
        print(f"\nTotal documents in collection: {self.collection.count()}")
        
        return stats


def main():
    """Main ingestion entry point."""
    print("="*60)
    print("LEGAL ASSISTANT RAG INGESTION PIPELINE")
    print("="*60)
    
    ingestor = LegalDocumentIngestor()
    stats = ingestor.ingest_all()
    
    print("\n✓ Ingestion complete!")


if __name__ == "__main__":
    main()
