# legal-rag-vibe

A Python-based Retrieval-Augmented-Generation (RAG) legal chatbot for Indian legal documents.

## Overview

**legal-rag-vibe** ingests authoritative Indian legal documents (statutes and judgments), indexes them into a local vector database (ChromaDB), and provides three primary capabilities:

1. **Legal Research Assistant** — Summaries, notes, and case studies from retrieved legal documents
2. **Judgment Simulation/Reference** — Hypothetical or real-case judgment-style analysis (clearly labeled as non-legal advice)
3. **Summarization/Headnote Generation** — Automated headnotes with facts, issues, holdings, ratio, and study notes

The system uses **Google AI Studio** for LLM generation (you must supply your free API key) and runs entirely locally for development and staging.

---

## Project Structure

```
legal-rag-vibe/
├── README.md
├── .env.example
├── requirements.txt
├── Dockerfile
├── makefile
├── data/
│   ├── raw/              # Raw PDFs and downloads
│   │   └── manifest.json # Download tracking manifest
│   ├── acts/             # Indian statutes (IPC, CrPC, etc.)
│   └── judgments/        # Court judgments
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── embeddings_.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── llm_client.py
│   ├── prompt_templates.py
│   ├── verify_and_log.py
│   ├── app_features.py
│   └── cli.py              # Command-line interface
├── scripts/
├── tests/
├── logs/                 # Request logs and audit files
└── chroma_db/           # ChromaDB persistence (created at runtime)
```

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Google AI Studio API key (free tier available at [aistudio.google.com](https://aistudio.google.com))
- Tesseract OCR installed (for scanned PDFs)
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
  - **macOS**: `brew install tesseract`
  - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your credentials:
   - **GOOGLE_API_KEY**: Paste your Google AI Studio API key
   - **GOOGLE_AI_ENDPOINT**: Set to your chosen model endpoint from Google AI Studio (e.g., `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent`)
   - Adjust other settings as needed (chunk size, port, etc.)

5. **Download legal documents**:
   
   Download Indian Supreme Court judgments from the AWS Open Data Registry:
   
   ```bash
   # Download 2023 judgments
   curl -o data/judgments/sc_2023_english.zip "https://indian-supreme-court-judgments.s3.amazonaws.com/data/zip/year=2023/english.zip"
   
   # Download 2024 judgments
   curl -o data/judgments/sc_2024_english.zip "https://indian-supreme-court-judgments.s3.amazonaws.com/data/zip/year=2024/english.zip"
   
   # Extract the zip files
   unzip data/judgments/sc_2023_english.zip -d data/judgments/
   unzip data/judgments/sc_2024_english.zip -d data/judgments/
   ```
   
   For more datasets, visit: https://github.com/vanga/indian-supreme-court-judgments

6. **Ingest documents into the vector database**:
   ```bash
   python -m src.ingest
   ```

7. **Run the CLI**:
   ```bash
   python -m src.cli
   ```


---

## Environment Variables

Copy `.env.example` to `.env` and configure the following:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | **Required**. Your Google AI Studio API key | (none) |
| `GOOGLE_AI_ENDPOINT` | **Required**. Full HTTP endpoint for your Google model | (none) |
| `CHROMA_DB_DIR` | Path to persist ChromaDB | `./chroma_db` |
| `EMBEDDING_MODEL` | Sentence-transformers model ID | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size for ingestion | `800` |
| `CHUNK_OVERLAP` | Overlap between chunks | `160` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

---

## Data Sources

This project uses **authoritative Indian legal sources**. Always respect robots.txt and terms of use.

### A. IndiaCode (indiacode.nic.in)

- **What**: Central Acts (IPC, CrPC, Evidence Act, etc.) in official PDF and HTML formats
- **How to use**: Download PDF versions of major Acts, save to `data/acts/`, and ingest into RAG
- **Usage**: Critical for statute lookups and authoritative definitions of sections

### B. Supreme Court of India (main.sci.gov.in/judgments)

- **What**: Official Supreme Court judgments and orders
- **How to use**: Download landmark judgments manually or use the downloader script to fetch PDF links from listing pages. Save to `data/judgments/`
- **Usage**: Primary source for precedents

### C. eCourts / High Court Judgments (judgments.ecourts.gov.in)

- **What**: High Court and some district court judgments
- **How to use**: Download selectively; be careful with rate-limits and TOS
- **Usage**: Add coverage for non-SC precedents

### D. GitHub & Open Data Dumps

- **What**: Community-maintained datasets and bulk dumps (e.g., Indian Supreme Court judgments on AWS Open Data)
- **How to use**: Use bulk downloads for fast bootstrap ingestion (unzip into `data/raw/`), then ingest
- **Usage**: Quick initial dataset

### E. IndianKanoon (indiankanoon.org)

- **What**: Unofficial index and search engine
- **How to use**: **Do not scrape at scale**. Use only for research to find case names, then download official PDFs
- **Usage**: Quick searches only; not for primary ingestion unless terms permit

---

## Usage Examples

### 1. Legal Research

Query the system for legal research on a topic:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "q": "What are the essential elements of Section 302 IPC?",
    "top_k": 6,
    "temperature": 0.0
  }'
```

**Response**:
```json
{
  "mode": "research",
  "answer": "Executive summary and bullet notes with citations [1], [2], etc.",
  "retrieved": [
    {
      "metadata": {
        "source_file": "IPC.pdf",
        "page_number": 142,
        "chunk_index": 5,
        "case_name": null,
        "court": null
      }
    }
  ],
  "verification": {
    "valid": [1, 2, 3],
    "invalid": []
  },
  "logfile": "logs/research_20231202_134905.json",
  "disclaimer": "For research/educational use only."
}
```

### 2. Judgment Simulation

Generate a hypothetical judgment analysis:

```bash
curl -X POST http://localhost:8000/judgment \
  -H "Content-Type: application/json" \
  -d '{
    "facts": "A person was found in possession of stolen goods. The prosecution alleges knowledge of theft.",
    "mode": "hypothetical",
    "top_k": 6,
    "temperature": 0.1
  }'
```

**Response**:
```json
{
  "mode": "judgment",
  "answer": "HYPOTHETICAL ANALYSIS — NOT LEGAL ADVICE\n\nFacts: ...\nIssues: ...\nReasoning: ...\nHypothetical Holding(s): ...\nSources: [1], [2], ...",
  "retrieved": [...],
  "verification": {
    "valid": [1, 2, 3, 4],
    "invalid": []
  },
  "logfile": "logs/judgment_20231202_135012.json",
  "disclaimer": "HYPOTHETICAL ANALYSIS — NOT LEGAL ADVICE"
}
```

### 3. Summarization / Headnote Generation

Summarize a judgment or generate headnotes:

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Kesavananda Bharati case",
    "top_k": 3,
    "temperature": 0.0
  }'
```

**Response**:
```json
{
  "mode": "summarize",
  "answer": "Facts: ...\nIssue: ...\nHolding: ...\nRatio: ...\nStudy Notes:\n1. ...\n2. ...",
  "logfile": "logs/summarize_20231202_135120.json"
}
```

---

## Ingestion Pipeline

The ingestion process (`src/ingest.py`) follows these steps:

1. **Discovery/Download**: Place PDFs into `data/raw/` using the downloader script
2. **Text Extraction**: Use `pdfplumber` to extract text; if empty, apply `pytesseract` OCR page-by-page
3. **Clean & Normalize**: Remove whitespace noise, preserve sections, parse headnotes/case-title/bench/date using regex heuristics
4. **Chunking**: Semantic chunking (600–1000 tokens per chunk) with 20–30% overlap, preserving page-level metadata
5. **Embedding**: Generate embeddings using local sentence-transformers model (`all-MiniLM-L6-v2`)
6. **Indexing**: Upsert into ChromaDB collection `legal_judgments` with metadata:
   - `source_file`, `page_number`, `chunk_index`, `case_name`, `court`, `judgement_date`, `url`
7. **Persist**: Save ChromaDB to `CHROMA_DB_DIR`

---

## API Endpoints

### POST /research

**Request**:
```json
{
  "q": "string (required)",
  "task": "string (optional)",
  "top_k": "int (optional, default 6)",
  "temperature": "float (optional, default 0.0)"
}
```

**Response**:
```json
{
  "mode": "research",
  "answer": "string",
  "retrieved": [{"metadata": {...}}],
  "verification": {"valid": [...], "invalid": [...]},
  "logfile": "string",
  "disclaimer": "For research/educational use only."
}
```

### POST /judgment

**Request**:
```json
{
  "facts": "string (required)",
  "mode": "hypothetical | reference (required)",
  "top_k": "int (optional, default 6)",
  "temperature": "float (optional, default 0.1)"
}
```

**Response**:
```json
{
  "mode": "judgment",
  "answer": "string (with header: HYPOTHETICAL ANALYSIS or REFERENCE ANALYSIS)",
  "retrieved": [{"metadata": {...}}],
  "verification": {"valid": [...], "invalid": [...]},
  "logfile": "string",
  "disclaimer": "string"
}
```

### POST /summarize

**Request**:
```json
{
  "query": "string (optional)",
  "case_text": "string (optional)",
  "top_k": "int (optional, default 3)",
  "temperature": "float (optional, default 0.0)"
}
```

**Response**:
```json
{
  "mode": "summarize",
  "answer": "string (Facts/Issue/Holding/Ratio + Study Notes)",
  "logfile": "string"
}
```

---

## Citation Verification & Anti-Hallucination

The system enforces strict citation policies to prevent hallucinations:

1. **Bracket-number citations**: All prompts instruct the model to cite using bracket numbers `[1]`, `[2]`, etc., corresponding to retrieved passages
2. **Post-generation verification**: Parse generated text for bracket citations and ensure every referenced number exists in the retrieved list
3. **Auto-retry on invalid citations**: If invalid citations are found, the system retries generation with stricter instructions (up to 2 retries)
4. **Unverified textual citations**: Case names or statutory citations outside bracketed references are flagged as "unverified" unless they match exact metadata entries

---

## Logging & Audit Trail

Every request is logged to `logs/` with the following information:

- Timestamp
- User ID (if present)
- Mode (research/judgment/summarize)
- User input (truncated)
- Retrieved metadata (IDs, file names, chunk indices)
- Assembled prompt (truncated)
- Model used and temperature
- LLM raw response (truncated)
- Verification result
- Path to log file

Logs are stored as JSON files and can be exported for human review using admin utilities.

---

## Docker Support

A `Dockerfile` is provided for containerized deployment:

```bash
docker build -t legal-rag-vibe .
docker run -p 8000:8000 --env-file .env legal-rag-vibe
```

---

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Tests include:
- Integration tests for API endpoints
- Unit tests for ingestion, retrieval, and verification modules
- Mock tests for LLM client

---

## Important Disclaimers

> [!CAUTION]
> **NOT LEGAL ADVICE**: This system is for research and educational purposes only. It does not provide legal advice. Always consult a qualified legal professional for legal matters.

> [!WARNING]
> **Data Source Compliance**: Always respect robots.txt and terms of use for data sources. Prefer official open datasets and IndiaCode. Do not scrape IndianKanoon or other unofficial sources at scale.

> [!IMPORTANT]
> **Citation Verification**: While the system attempts to verify citations, always manually verify critical legal references against official sources.

> [!NOTE]
> **API Key Security**: Never commit your `.env` file or expose your Google API key publicly. Use environment variables or secret management systems in production.

---

## License

This project is provided as-is for educational and research purposes. Ensure compliance with all applicable laws and terms of service when using legal data sources.

---

## Contributing

Contributions are welcome! Please ensure all code follows the project structure and includes appropriate tests and documentation.

---

## Support

For issues or questions, please open an issue on the project repository.
