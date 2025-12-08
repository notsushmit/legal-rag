"""
LLM client for Legal Assistant RAG Chatbot.

Calls Google AI Studio generative API with structured request/response parsing
and safe fallbacks. Handles prompt formatting, generation parameters, and
error handling with retries.
"""

import json
import time
from typing import Dict, Optional, Any
import requests

from src.config import Config


class GoogleAIClient:
    """Client for Google AI Studio API."""
    
    def __init__(self):
        """Initialize the Google AI client."""
        self.api_key = Config.GOOGLE_API_KEY
        self.endpoint = Config.GOOGLE_AI_ENDPOINT
        
        if not self.api_key or not self.endpoint:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_AI_ENDPOINT must be set in .env")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = None,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generate text using Google AI Studio API.
        
        Args:
            prompt: The prompt text
            temperature: Generation temperature (0.0-1.0)
            max_output_tokens: Maximum tokens to generate
            retry_count: Number of retries on failure
        
        Returns:
            Dict with 'text' and 'raw_response' keys
        """
        if max_output_tokens is None:
            max_output_tokens = Config.MAX_OUTPUT_TOKENS
        
        # Prepare request payload
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key to URL
        url = f"{self.endpoint}?key={self.api_key}"
        
        # Retry logic
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract text from response
                    text = self._extract_text(result)
                    
                    return {
                        "text": text,
                        "raw_response": result
                    }
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    print(f"Attempt {attempt + 1}/{retry_count} failed: {error_msg}")
                    
                    if attempt < retry_count - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return {
                            "text": f"[ERROR: {error_msg}]",
                            "raw_response": {"error": error_msg}
                        }
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retry_count} failed: {e}")
                
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "text": f"[ERROR: {str(e)}]",
                        "raw_response": {"error": str(e)}
                    }
        
        return {
            "text": "[ERROR: All retry attempts failed]",
            "raw_response": {"error": "All retry attempts failed"}
        }
    
    def _extract_text(self, response: Dict) -> str:
        """
        Extract generated text from Google AI response.
        
        Args:
            response: Raw API response
        
        Returns:
            Generated text string
        """
        try:
            # Google AI Studio response format
            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            
            # Fallback: stringify the response
            return json.dumps(response, indent=2)
        
        except Exception as e:
            print(f"Error extracting text from response: {e}")
            return json.dumps(response, indent=2)


# Global instance
_llm_client_instance = None


def get_llm_client() -> GoogleAIClient:
    """Get or create the global LLM client instance."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = GoogleAIClient()
    return _llm_client_instance
