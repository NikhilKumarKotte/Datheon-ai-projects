"""Local semantic memory backed by ChromaDB and sentence-transformers.

Each chat is isolated by conversation_id metadata. The embedding model runs locally
after its first download; no conversation text is sent to an embedding API.
"""
import uuid
from typing import List

MEMORY_AVAILABLE = True
MEMORY_ERROR = None

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - depends on local environment
    MEMORY_AVAILABLE = False
    MEMORY_ERROR = str(exc)


class SemanticMemory:
    def __init__(self, persist_directory: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not MEMORY_AVAILABLE:
            raise RuntimeError(MEMORY_ERROR or "Semantic memory dependencies unavailable")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="datheon_conversation_memory",
            metadata={"hnsw:space": "cosine"}
        )
        self.model = SentenceTransformer(model_name)

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        # Avoid indexing very large transcripts as one vector.
        text = text[:3500]
        vector = self.model.encode(text, normalize_embeddings=True).tolist()
        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=[vector],
            metadatas=[{"conversation_id": conversation_id, "role": role}]
        )

    def search(self, conversation_id: str, query: str, limit: int = 4) -> List[str]:
        query = (query or "").strip()
        if not query:
            return []
        vector = self.model.encode(query, normalize_embeddings=True).tolist()
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(max(limit * 4, limit), count),
            where={"conversation_id": conversation_id},
            include=["documents", "distances", "metadatas"]
        )
        docs = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        matches = []
        for doc, distance in zip(docs, distances):
            # Cosine distance threshold avoids injecting weakly related memories.
            if doc and distance <= 0.65:
                matches.append(doc)
            if len(matches) >= limit:
                break
        return matches
