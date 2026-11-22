from langchain_core.documents import Document
from src.rag_pipeline import RAGPipeline


def test_explain_methodology_uses_generation_not_full_extraction():
    assert RAGPipeline._requested_section(
        "Explain the methodology of the project in detail"
    ) is None


def test_explain_architecture_uses_generation_not_full_extraction():
    assert RAGPipeline._requested_section(
        "Explain the complete system architecture, including every component"
    ) is None


def test_full_toc_request_is_extraction():
    assert RAGPipeline._requested_section(
        "Show me the complete table of contents exactly as provided in the uploaded PDF"
    ) == "table of contents"


def test_short_give_abstract_and_toc_remain_supported():
    assert RAGPipeline._requested_section("give abstract") == "abstract"
    assert RAGPipeline._requested_section("give TOC") == "table of contents"
    assert RAGPipeline._requested_section("give content") == "table of contents"


def test_unsupported_annual_revenue_is_rejected_without_numeric_evidence():
    docs = [Document(
        page_content="The system uses retrieval augmented generation and vector embeddings."
    )]
    ok, reason = RAGPipeline._query_evidence_check(
        "What is the annual revenue of the company?", docs
    )
    assert not ok
    assert "meaningful terms" in reason or "revenue" in reason


def test_architecture_question_passes_when_relevant_terms_are_present():
    docs = [Document(
        page_content=(
            "The proposed system architecture contains a retrieval module, "
            "an agent controller, and a response generation component."
        )
    )]
    ok, _ = RAGPipeline._query_evidence_check(
        "Explain the complete system architecture and components", docs
    )
    assert ok
