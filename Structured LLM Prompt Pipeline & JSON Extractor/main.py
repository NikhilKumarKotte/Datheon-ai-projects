"""
Structured LLM Document Extractor
Root entrypoint delegating to src.extractor.cli for CLI, web server, and extraction services.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path to allow running directly from repository root
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.extractor.api.app import app
from src.extractor.cli import main
from src.extractor.core.service import extract_document
from src.extractor.schemas.invoice import InvoiceData, InvoiceLineItem
from src.extractor.schemas.receipt import ReceiptData, ReceiptLineItem
from src.extractor.schemas.resume import ResumeData

__all__ = [
    "app",
    "main",
    "extract_document",
    "InvoiceData",
    "InvoiceLineItem",
    "ReceiptData",
    "ReceiptLineItem",
    "ResumeData",
]

if __name__ == "__main__":
    main()
