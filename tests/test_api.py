"""
Integration and unit tests for legal-rag-vibe.

Tests cover:
- API endpoints
- Ingestion pipeline
- Retrieval functionality
- Citation verification
- LLM client
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# Import app
from src.app_features import app
from src.verify_and_log import verify_bracket_citations


# Test client
client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "legal-rag-vibe"
    assert "endpoints" in data


def test_research_endpoint_missing_query():
    """Test research endpoint with missing query."""
    response = client.post("/research", json={})
    assert response.status_code == 422  # Validation error


def test_judgment_endpoint_invalid_mode():
    """Test judgment endpoint with invalid mode."""
    response = client.post("/judgment", json={
        "facts": "Test facts",
        "mode": "invalid_mode"
    })
    assert response.status_code == 400


def test_summarize_endpoint_missing_input():
    """Test summarize endpoint with missing input."""
    response = client.post("/summarize", json={})
    assert response.status_code == 400


def test_citation_verification_valid():
    """Test citation verification with valid citations."""
    text = "According to [1] and [2], the law states..."
    result = verify_bracket_citations(text, num_retrieved=3)
    
    assert result["valid"] == [1, 2]
    assert result["invalid"] == []


def test_citation_verification_invalid():
    """Test citation verification with invalid citations."""
    text = "According to [1] and [5], the law states..."
    result = verify_bracket_citations(text, num_retrieved=3)
    
    assert result["valid"] == [1]
    assert result["invalid"] == [5]


def test_citation_verification_no_citations():
    """Test citation verification with no citations."""
    text = "This text has no citations."
    result = verify_bracket_citations(text, num_retrieved=3)
    
    assert result["valid"] == []
    assert result["invalid"] == []


# Add more integration tests as needed
# These would require a populated ChromaDB and valid API credentials

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
