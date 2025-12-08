"""
Configuration module for Legal Assistant RAG Chatbot.

Reads environment variables and provides centralized configuration
for all modules including Google AI Studio credentials, ChromaDB settings,
embedding model configuration, and server parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the application."""
    
    # Google AI Studio Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_AI_ENDPOINT: str = os.getenv("GOOGLE_AI_ENDPOINT", "")
    
    # ChromaDB Configuration
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = "legal_judgments"
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "160"))
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    ACTS_DIR: Path = DATA_DIR / "acts"
    JUDGMENTS_DIR: Path = DATA_DIR / "judgments"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Retrieval Configuration
    DEFAULT_TOP_K: int = 6
    RERANK_TOP_K: int = 50
    
    # Generation Configuration
    RESEARCH_TEMPERATURE: float = 0.0
    JUDGMENT_TEMPERATURE: float = 0.1
    SUMMARIZE_TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 2048
    
    # Download Configuration
    DOWNLOAD_DELAY: float = 1.0  # seconds between requests
    USER_AGENT: str = "legal-assistant-rag/1.0 (Educational Research Tool)"
    MAX_RETRIES: int = 3
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if not cls.GOOGLE_API_KEY:
            print("ERROR: GOOGLE_API_KEY not set in .env file")
            return False
        
        if not cls.GOOGLE_AI_ENDPOINT:
            print("ERROR: GOOGLE_AI_ENDPOINT not set in .env file")
            return False
        
        return True
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.ACTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        Path(cls.CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)


# Validate configuration on import
if not Config.validate():
    print("WARNING: Configuration validation failed. Please check your .env file.")
