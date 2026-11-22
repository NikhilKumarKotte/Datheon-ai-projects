# Abstract / Section Retrieval Fix

Replace `src/vector_store.py`, `src/rag_pipeline.py`, and `app.py` with these
versions. This is a retrieval-only patch; no vector database reset is required.

The previous flow relied on dense retrieval alone. A short request such as
"give abstract" could rank a semantically adjacent but incorrect page above the
page whose heading is exactly ABSTRACT. The local extractive answerer then
searched those wrong passages for the literal word "abstract" and refused.

This patch adds lexical + dense hybrid retrieval. For section requests such as
abstract, summary, conclusion, objectives, problem statement, and methodology,
it locates the page containing the section heading and retrieves all chunks
from that same source page. The local answerer merges overlapping chunks to
reconstruct the section.

Scores are ranking signals, not calibrated probabilities or proof of factuality.
