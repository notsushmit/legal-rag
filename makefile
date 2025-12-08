.PHONY: help venv install dev test clean ingest download

help:
	@echo "Legal Assistant RAG Chatbot Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  venv       - Create virtual environment"
	@echo "  install    - Install dependencies"
	@echo "  dev        - Run development server"
	@echo "  test       - Run tests"
	@echo "  ingest     - Run ingestion pipeline"
	@echo "  download   - Run download script (interactive)"
	@echo "  clean      - Clean temporary files"

venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

install:
	pip install -r requirements.txt

dev:
	uvicorn src.app_features:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

.PHONY: help venv install dev test clean ingest download run-cli

help:
	@echo "Legal Assistant RAG Chatbot Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  venv       - Create virtual environment"
	@echo "  install    - Install dependencies"
	@echo "  dev        - Run development server"
	@echo "  test       - Run tests"
	@echo "  ingest     - Run ingestion pipeline"
	@echo "  download   - Run download script (interactive)"
	@echo "  run-cli    - Run the command-line interface"
	@echo "  clean      - Clean temporary files"

venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

install:
	pip install -r requirements.txt

dev:
	uvicorn src.app_features:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

ingest:
	python -m src.ingest

download:
	python scripts/download_datasets.py

run-cli:
	$(PYTHON) -m src.cli

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
