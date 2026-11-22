"""
Vector store management module supporting both ChromaDB and FAISS.
Provides unified indexing, persistence, retrieval, and similarity scoring.
"""

import os
import shutil
import logging
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import CHROMA_PERSIST_DIR, FAISS_PERSIST_DIR, default_config
from src.embeddings import EmbeddingFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_chroma_class():
    """Dynamically loads the preferred Chroma class."""
    try:
        from langchain_chroma import Chroma
        return Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
        return Chroma


def get_faiss_class():
    """Loads the FAISS vectorstore class."""
    from langchain_community.vectorstores import FAISS
    return FAISS


class VectorStoreManager:
    """Unified manager for ChromaDB and FAISS vector indices."""

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        faiss_dir: Optional[Path] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.chroma_dir = Path(chroma_dir or CHROMA_PERSIST_DIR)
        self.faiss_dir = Path(faiss_dir or FAISS_PERSIST_DIR)
        self._embeddings = embeddings

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

    def _embedding_signature(self) -> str:
        """Stable best-effort identity for the embedding model used by this manager."""
        emb = self.embeddings
        model = (
            getattr(emb, "model_name", None)
            or getattr(emb, "model", None)
            or emb.__class__.__name__
        )
        if not isinstance(model, str):
            model = str(model)
        return f"{emb.__class__.__module__}.{emb.__class__.__name__}:{model}"

    def _validate_embedding_manifest(self, store_dir: Path) -> None:
        manifest = store_dir / "embedding_manifest.json"
        current = self._embedding_signature()
        if not manifest.exists():
            raise RuntimeError(
                f"Index at {store_dir} has no embedding-model manifest. Rebuild this index once "
                "with the currently selected embedding model to prevent mixed-vector retrieval."
            )
        saved = json.loads(manifest.read_text(encoding="utf-8")).get("embedding_signature")
        if saved != current:
            raise RuntimeError(
                "The selected embedding model does not match the model used to build this index. "
                "Select the original model or reset and re-index all documents."
            )

    def _write_embedding_manifest(self, store_dir: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        payload = {"embedding_signature": self._embedding_signature(), "format_version": 1}
        (store_dir / "embedding_manifest.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @property
    def embeddings(self) -> Embeddings:
        """Lazily obtains embeddings if not explicitly injected."""
        if self._embeddings is None:
            self._embeddings = EmbeddingFactory.get_embeddings(
                provider=default_config.embedding_provider,
                model_name=default_config.embedding_model_name,
            )
        return self._embeddings

    def create_or_add_documents(
        self,
        documents: List[Document],
        store_type: str = "chroma",
        collection_name: str = "document_qa_collection",
    ) -> Any:
        """Indexes documents into the selected vector store and persists to disk.

        Args:
            documents: List of chunked Document instances.
            store_type: 'chroma' or 'faiss'.
            collection_name: Collection name for ChromaDB.

        Returns:
            The instantiated vector store.
        """
        if not documents:
            raise ValueError("No documents provided to index into vector store.")

        store_type = store_type.lower().strip()
        logger.info(
            f"Indexing {len(documents)} chunks into {store_type.upper()} vector store..."
        )

        if store_type == "chroma":
            ChromaClass = get_chroma_class()
            manifest = self.chroma_dir / "embedding_manifest.json"
            existing_entries = [p for p in self.chroma_dir.iterdir() if p.name != "embedding_manifest.json"]
            if existing_entries and not manifest.exists():
                raise RuntimeError(
                    "Existing ChromaDB data has no embedding-model manifest. Reset this old index "
                    "once, then re-index all documents with the selected embedding model."
                )
            if existing_entries and manifest.exists():
                self._validate_embedding_manifest(self.chroma_dir)
                vector_store = ChromaClass(
                    persist_directory=str(self.chroma_dir),
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                )
                ids = [str((doc.metadata or {}).get("chunk_id") or __import__("uuid").uuid4()) for doc in documents]
                # Chroma upserts by ID, making repeated ingestion of the same
                # filename/page/content idempotent instead of duplicating vectors.
                vector_store.add_documents(documents=documents, ids=ids)
            else:
                ids = [str((doc.metadata or {}).get("chunk_id") or __import__("uuid").uuid4()) for doc in documents]
                vector_store = ChromaClass.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    ids=ids,
                    persist_directory=str(self.chroma_dir),
                    collection_name=collection_name,
                )
            self._write_embedding_manifest(self.chroma_dir)
            logger.info(f"Successfully persisted ChromaDB index at '{self.chroma_dir}'.")
            return vector_store

        elif store_type == "faiss":
            FAISSClass = get_faiss_class()
            # If an existing FAISS index exists on disk, append to it; otherwise create new
            index_file = self.faiss_dir / "index.faiss"
            if index_file.exists():
                try:
                    self._validate_embedding_manifest(self.faiss_dir)
                    existing_store = FAISSClass.load_local(
                        folder_path=str(self.faiss_dir),
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    known_ids = set(existing_store.index_to_docstore_id.values())
                    new_docs = [d for d in documents if str((d.metadata or {}).get("chunk_id")) not in known_ids]
                    if new_docs:
                        ids = [str((d.metadata or {}).get("chunk_id") or __import__("uuid").uuid4()) for d in new_docs]
                        existing_store.add_documents(new_docs, ids=ids)
                    existing_store.save_local(str(self.faiss_dir))
                    self._write_embedding_manifest(self.faiss_dir)
                    logger.info(f"Appended {len(documents)} chunks to existing FAISS index.")
                    return existing_store
                except Exception as e:
                    logger.error(f"Could not safely append to FAISS index: {e}")
                    raise RuntimeError(
                        "Could not safely update the existing FAISS index. Check its embedding manifest "
                        "or reset and re-index it; the old index was not intentionally overwritten."
                    ) from e

            ids = [str((doc.metadata or {}).get("chunk_id") or __import__("uuid").uuid4()) for doc in documents]
            vector_store = FAISSClass.from_documents(
                documents=documents,
                embedding=self.embeddings,
                ids=ids,
            )
            vector_store.save_local(str(self.faiss_dir))
            self._write_embedding_manifest(self.faiss_dir)
            logger.info(f"Successfully persisted FAISS index at '{self.faiss_dir}'.")
            return vector_store

        else:
            raise ValueError(f"Unsupported vector store type: '{store_type}'. Choose 'chroma' or 'faiss'.")

    def load_store(
        self,
        store_type: str = "chroma",
        collection_name: str = "document_qa_collection",
    ) -> Optional[Any]:
        """Loads an existing persisted vector store from disk.

        Args:
            store_type: 'chroma' or 'faiss'.
            collection_name: Collection name for ChromaDB.

        Returns:
            Instantiated vector store or None if not found/empty.
        """
        store_type = store_type.lower().strip()

        if store_type == "chroma":
            # Check if directory has contents
            if not any(self.chroma_dir.iterdir()):
                logger.info(f"No existing ChromaDB index found at '{self.chroma_dir}'.")
                return None
            self._validate_embedding_manifest(self.chroma_dir)
            try:
                ChromaClass = get_chroma_class()
                store = ChromaClass(
                    persist_directory=str(self.chroma_dir),
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                )
                # Verify collection is non-empty
                count = store._collection.count() if hasattr(store, "_collection") else 1
                if count == 0:
                    logger.info("ChromaDB collection exists but is currently empty.")
                    return None
                return store
            except Exception as e:
                logger.warning(f"Error loading ChromaDB store: {e}")
                return None

        elif store_type == "faiss":
            index_file = self.faiss_dir / "index.faiss"
            if not index_file.exists():
                logger.info(f"No existing FAISS index file found at '{index_file}'.")
                return None
            self._validate_embedding_manifest(self.faiss_dir)
            try:
                FAISSClass = get_faiss_class()
                store = FAISSClass.load_local(
                    folder_path=str(self.faiss_dir),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return store
            except Exception as e:
                logger.warning(f"Error loading FAISS store: {e}")
                return None

        else:
            raise ValueError(f"Unknown store type: '{store_type}'")

    def as_retriever(
        self,
        store_type: str = "chroma",
        k: int = 4,
        search_type: str = "similarity",
    ) -> VectorStoreRetriever:
        """Returns a LangChain retriever for the given vector store."""
        store = self.load_store(store_type=store_type)
        if store is None:
            raise RuntimeError(
                f"Cannot create retriever: {store_type.upper()} index does not exist. "
                "Please load and index documents first."
            )
        return store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )


    @staticmethod
    def _lexical_tokens(text: str) -> List[str]:
        """Tokenize text for a lightweight lexical retrieval pass."""
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
            "by", "can", "could", "did", "do", "does", "for", "from", "give",
            "had", "has", "have", "how", "i", "in", "into", "is", "it", "me",
            "of", "on", "or", "our", "please", "tell", "that", "the", "their",
            "them", "there", "these", "this", "to", "was", "we", "were", "what",
            "when", "where", "which", "who", "why", "will", "with", "would",
            "you", "your"
        }
        return [
            token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if token not in stop and len(token) > 1
        ]

    def _get_all_indexed_documents(self, store: Any, store_type: str) -> List[Document]:
        """Read indexed document text/metadata for the lexical retrieval pass."""
        docs: List[Document] = []
        if store_type == "chroma":
            payload = store.get(include=["documents", "metadatas"])
            for text, metadata in zip(payload.get("documents") or [], payload.get("metadatas") or []):
                if text:
                    docs.append(Document(page_content=text, metadata=metadata or {}))
        elif store_type == "faiss":
            docstore = getattr(store, "docstore", None)
            mapping = getattr(docstore, "_dict", {}) if docstore is not None else {}
            docs.extend(
                item for item in mapping.values()
                if isinstance(item, Document) and item.page_content
            )
        return docs

    @staticmethod
    def _chunk_order(doc: Document) -> tuple:
        """Stable ordering for chunks from the same source page."""
        chunk_id = str((doc.metadata or {}).get("chunk_id", ""))
        match = re.search(r"_c(\d+)$", chunk_id)
        return (int(match.group(1)) if match else 0, chunk_id)

    def hybrid_search_with_confidence(
        self, query: str, k: int = 4, store_type: str = "chroma"
    ) -> List[Tuple[Document, float]]:
        """
        Combine dense and lexical retrieval. Section-heading requests retrieve all
        chunks from the page containing that heading, e.g. "give abstract".
        Returned scores are ranking signals, not calibrated probabilities.
        """
        store_type = store_type.lower().strip()
        store = self.load_store(store_type=store_type)
        if store is None:
            return []

        dense = self.similarity_search_with_confidence(
            query=query, k=max(k * 4, 16), store_type=store_type
        )
        all_docs = self._get_all_indexed_documents(store, store_type)
        if not all_docs:
            return dense[:k]

        query_tokens = self._lexical_tokens(query)
        if not query_tokens:
            return dense[:k]
        query_set = set(query_tokens)

        raw_query = (query or "").lower()
        extraction_intent = bool(re.search(
            r"\b(show|display|extract|print|reproduce|quote|copy|list)\b", raw_query
        )) or bool(re.search(
            r"\b(give|provide)\s+(?:me\s+)?(?:the\s+)?(?:full|complete|entire|exact)\b", raw_query
        )) or bool(re.search(
            r"\bexactly\s+as\s+(?:provided|written|shown|presented)\b", raw_query
        )) or bool(re.match(
            r"^\s*(?:please\s+)?give\s+(?:me\s+)?(?:the\s+)?"
            r"(abstract|objectives?|summary|conclusion|methodology|problem statement|toc|contents?)\s*$", raw_query
        ))
        section_aliases = {
            "table of contents": ("table of contents",),
            "abstract": ("abstract",),
            "summary": ("summary", "executive summary"),
            "conclusion": ("conclusion", "overall conclusion"),
            "objectives": ("objectives", "objective"),
            "problem statement": ("problem statement",),
            "methodology": ("proposed methodology", "methodology"),
        }
        requested_section = None
        if extraction_intent and (
            re.search(r"\btable\s+of\s+contents?\b", raw_query)
            or re.search(r"\btoc\b", raw_query)
            or re.search(r"\b(show|give|display|list|extract|print)\s+(?:me\s+)?(?:the\s+)?contents?\b", raw_query)
            or re.search(r"\bcomplete\s+contents?\b", raw_query)
        ):
            requested_section = "table of contents"
        elif extraction_intent:
            query_text = " ".join(query_tokens)
            requested_section = next(
                (name for name in section_aliases if name != "table of contents" and name in query_text),
                None,
            )

        if requested_section:
            aliases = section_aliases[requested_section]
            heading_hits = []
            for doc in all_docs:
                lines = [
                    re.sub(r"\s+", " ", line).strip(" .:-").lower()
                    for line in (doc.page_content or "").splitlines()
                    if line.strip()
                ]
                for line in lines:
                    if any(line == alias or line == f"{alias} (continued)" for alias in aliases):
                        heading_hits.append(doc)
                        break

            if heading_hits:
                source_keys = {
                    (d.metadata or {}).get("file_name") or (d.metadata or {}).get("source")
                    for d in heading_hits
                }
                target_pages = {
                    (
                        (d.metadata or {}).get("file_name") or (d.metadata or {}).get("source"),
                        int((d.metadata or {}).get("page", 1) or 1),
                    )
                    for d in heading_hits
                }

                # Include every immediately following page explicitly marked as
                # a continuation of the Table of Contents.
                if requested_section == "table of contents":
                    for source_key in source_keys:
                        source_docs = [
                            d for d in all_docs
                            if ((d.metadata or {}).get("file_name") or (d.metadata or {}).get("source")) == source_key
                        ]
                        page_map = {}
                        for d in source_docs:
                            pno = int((d.metadata or {}).get("page", 1) or 1)
                            page_map.setdefault(pno, []).append(d)

                        starts = sorted(p for src, p in target_pages if src == source_key)
                        for first_page in starts:
                            next_page = first_page + 1
                            while next_page in page_map:
                                next_text = "\n".join(x.page_content for x in page_map[next_page]).lower()
                                normalized = re.sub(r"\s+", " ", next_text)
                                if "table of contents (continued)" not in normalized:
                                    break
                                target_pages.add((source_key, next_page))
                                next_page += 1

                page_docs = [
                    d for d in all_docs
                    if (
                        (d.metadata or {}).get("file_name") or (d.metadata or {}).get("source"),
                        int((d.metadata or {}).get("page", 1) or 1),
                    ) in target_pages
                ]
                page_docs.sort(key=lambda d: (
                    str((d.metadata or {}).get("file_name", "")),
                    int((d.metadata or {}).get("page", 1) or 1),
                    int((d.metadata or {}).get("chunk_index", 0) or 0),
                ))
                return [(doc, 1.0) for doc in page_docs]

        lexical_ranked = []
        for doc in all_docs:
            tokens = self._lexical_tokens(doc.page_content)
            matched = query_set.intersection(tokens)
            if not matched:
                continue
            coverage = len(matched) / max(1, len(query_set))
            tf_bonus = min(0.15, sum(min(tokens.count(t), 3) for t in matched) * 0.025)
            lexical_ranked.append((doc, min(1.0, coverage + tf_bonus)))
        lexical_ranked.sort(key=lambda item: item[1], reverse=True)

        dense_map = {
            str((d.metadata or {}).get("chunk_id") or d.page_content[:120]): score
            for d, score in dense
        }
        lexical_map = {
            str((d.metadata or {}).get("chunk_id") or d.page_content[:120]): score
            for d, score in lexical_ranked[:max(k * 4, 16)]
        }
        docs_by_key = {}
        for doc, _ in dense + lexical_ranked[:max(k * 4, 16)]:
            key = str((doc.metadata or {}).get("chunk_id") or doc.page_content[:120])
            docs_by_key.setdefault(key, doc)

        combined = []
        for key, doc in docs_by_key.items():
            score = 0.40 * dense_map.get(key, 0.0) + 0.60 * lexical_map.get(key, 0.0)
            combined.append((doc, score))
        combined.sort(key=lambda item: item[1], reverse=True)
        return combined[:k]

    def similarity_search_with_confidence(
        self,
        query: str,
        k: int = 4,
        store_type: str = "chroma",
    ) -> List[Tuple[Document, float]]:
        """Performs similarity search and returns a monotonic retrieval score proxy.

        Args:
            query: The question or query string.
            k: Number of chunks to retrieve.
            store_type: 'chroma' or 'faiss'.

        Returns:
            List of (Document, monotonic_retrieval_score) between 0.0 and 1.0.
            This is a distance transform for ranking/filtering, NOT a calibrated
            probability that the retrieved chunk answers the question.
        """
        store = self.load_store(store_type=store_type)
        if store is None:
            return []

        results_with_scores = store.similarity_search_with_score(query, k=k)
        normalized: List[Tuple[Document, float]] = []

        for doc, raw_score in results_with_scores:
            # Score normalization:
            # Chroma returns L2 squared distance (lower is closer; 0 is exact match) or cosine distance.
            # FAISS returns L2 distance by default.
            # Convert distance d to confidence score: s = 1.0 / (1.0 + d) or max(0, 1 - d/2)
            try:
                score_val = float(raw_score)
                if score_val >= 0:
                    confidence = round(1.0 / (1.0 + score_val), 4)
                else:
                    # Cosine similarity in [-1, 1]
                    confidence = round(max(0.0, min(1.0, (score_val + 1.0) / 2.0)), 4)
            except (TypeError, ValueError) as exc:
                # Never fabricate a high score when a backend returns an
                # unexpected score type; fail visibly instead.
                raise RuntimeError(f"Vector store returned an invalid distance score: {raw_score!r}") from exc

            normalized.append((doc, confidence))

        return normalized

    def delete_document(self, file_name: str, store_type: str = "chroma") -> int:
        """Delete all indexed chunks belonging to a filename; return deleted count."""
        if not file_name or not file_name.strip():
            raise ValueError("file_name is required")
        store_type = store_type.lower().strip()
        store = self.load_store(store_type=store_type)
        if store is None:
            return 0

        if store_type == "chroma":
            matches = store.get(where={"file_name": file_name}, include=[])
            ids = matches.get("ids", []) if matches else []
            if ids:
                store.delete(ids=ids)
            return len(ids)

        if store_type == "faiss":
            ids = []
            for doc_id, doc in store.docstore._dict.items():
                metadata = getattr(doc, "metadata", {}) or {}
                if metadata.get("file_name") == file_name:
                    ids.append(doc_id)
            if ids:
                store.delete(ids=ids)
                store.save_local(str(self.faiss_dir))
            return len(ids)

        raise ValueError(f"Unsupported vector store type: {store_type}")

    def clear_store(self, store_type: str = "chroma") -> bool:
        """Clears/resets the persisted index on disk."""
        target_dir = self.chroma_dir if store_type.lower() == "chroma" else self.faiss_dir
        logger.info(f"Clearing vector store directory: {target_dir}")
        try:
            if target_dir.exists():
                shutil.rmtree(target_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store at {target_dir}: {e}")
            return False

    def get_stats(self, store_type: str = "chroma") -> Dict[str, Any]:
        """Returns statistics regarding index status and vector counts."""
        store_type = store_type.lower().strip()
        store = self.load_store(store_type=store_type)

        if store is None:
            return {
                "store_type": store_type,
                "is_indexed": False,
                "vector_count": 0,
                "path": str(self.chroma_dir if store_type == "chroma" else self.faiss_dir),
            }

        count = 0
        try:
            if store_type == "chroma":
                count = store._collection.count()
            elif store_type == "faiss":
                count = store.index.ntotal
        except Exception:
            count = "Active"

        return {
            "store_type": store_type,
            "is_indexed": True,
            "vector_count": count,
            "path": str(self.chroma_dir if store_type == "chroma" else self.faiss_dir),
        }
