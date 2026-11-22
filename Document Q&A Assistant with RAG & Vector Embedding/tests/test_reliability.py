from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.document_loader import DocumentLoaderService
from src.text_splitter import TextChunkerService


def test_upload_rejects_path_traversal_and_stores_safe_unique_name(tmp_path):
    # Text files are accepted, but the supplied path is reduced to a basename.
    docs = DocumentLoaderService.save_and_load_uploaded_file(
        b"A sufficiently long document sentence for testing upload safety.",
        "../../unsafe name.txt", tmp_path
    )
    assert docs
    assert docs[0].metadata["file_name"] == "unsafe name.txt"
    assert Path(docs[0].metadata["source"]).parent == tmp_path.resolve()
    assert ".." not in Path(docs[0].metadata["stored_file_name"]).name


def test_upload_rejects_empty_and_unsupported_files(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        DocumentLoaderService.save_and_load_uploaded_file(b"", "empty.txt", tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        DocumentLoaderService.save_and_load_uploaded_file(b"data", "payload.exe", tmp_path)


def test_upload_rejects_invalid_pdf_header(tmp_path):
    with pytest.raises(ValueError, match="PDF header"):
        DocumentLoaderService.save_and_load_uploaded_file(b"not a pdf", "fake.pdf", tmp_path)


def test_chunk_ids_are_stable_for_same_filename_page_and_content():
    chunker = TextChunkerService(chunk_size=100, chunk_overlap=10)
    text = "This is a stable test paragraph with enough content to generate a document chunk. " * 2
    first = chunker.split_documents([Document(page_content=text, metadata={"file_name": "a.txt", "page": 1})])
    second = chunker.split_documents([Document(page_content=text, metadata={"file_name": "a.txt", "page": 1})])
    assert [d.metadata["chunk_id"] for d in first] == [d.metadata["chunk_id"] for d in second]
