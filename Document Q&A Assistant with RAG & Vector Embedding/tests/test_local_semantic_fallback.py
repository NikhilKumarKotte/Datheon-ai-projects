from langchain_core.documents import Document
from src.rag_pipeline import RAGPipeline


def test_local_mode_returns_evidence_when_query_uses_synonyms():
    pipeline = RAGPipeline()
    retrieved = [(
        Document(
            page_content=(
                "Employees are entitled to paid time off each year. "
                "Unused leave is handled according to the annual leave policy."
            ),
            metadata={"file_name": "handbook.pdf", "page": 4},
        ),
        0.62,
    )]

    answer = pipeline._synthesize_local_response(
        "What is the company's leave policy?", retrieved
    )

    assert "paid time off" in answer.lower() or "annual leave" in answer.lower()
    assert "handbook.pdf" in answer
    assert "Page: 4" in answer
    assert "do not contain sufficient information" not in answer.lower()


def test_local_mode_refuses_when_retrieval_signal_is_very_low():
    pipeline = RAGPipeline()
    retrieved = [(
        Document(
            page_content="The office cafeteria serves lunch from noon.",
            metadata={"file_name": "handbook.pdf", "page": 2},
        ),
        0.08,
    )]

    answer = pipeline._synthesize_local_response(
        "What is the company's leave policy?", retrieved
    )

    assert "could not find" in answer.lower()
