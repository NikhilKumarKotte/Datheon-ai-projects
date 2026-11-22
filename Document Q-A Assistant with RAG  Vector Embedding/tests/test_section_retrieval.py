from langchain_core.documents import Document
from src.rag_pipeline import RAGPipeline

def test_section_intent_detects_abstract():
    assert RAGPipeline._requested_section("give abstract") == "abstract"

def test_section_intent_detects_conclusion():
    assert RAGPipeline._requested_section("show me the conclusion") == "conclusion"

def test_overlapping_chunks_merge_without_repeating_overlap():
    docs = [
        Document(page_content="The abstract begins with a project overview and describes the"),
        Document(page_content="describes the proposed system and its objectives."),
    ]
    assert RAGPipeline._merge_chunk_text(docs) == (
        "The abstract begins with a project overview and describes the "
        "proposed system and its objectives."
    )


def test_section_intent_detects_table_of_contents():
    assert RAGPipeline._requested_section("show the complete table of contents") == "table of contents"
    assert RAGPipeline._requested_section("give TOC") == "table of contents"
    assert RAGPipeline._requested_section("give content") == "table of contents"


def test_chunk_merge_respects_chunk_index_order():
    docs = [
        Document(page_content="second chunk ends here.", metadata={"chunk_index": 1, "page": 6}),
        Document(page_content="first chunk begins here.", metadata={"chunk_index": 0, "page": 6}),
    ]
    assert RAGPipeline._merge_chunk_text(docs).startswith("first chunk begins here.")
