STREAMLIT CLOUD DEPENDENCY FILE FIX

The supplied project archive has a valid requirements.txt inside:
Document Q&A Assistant with RAG & Vector Embeddings/requirements.txt

The Streamlit Cloud log shows it is trying to parse `Q&A` as a package requirement.
That indicates Streamlit Cloud is reading a wrong dependency-file path or a malformed
root-level dependency file, not the valid nested requirements file in the supplied ZIP.

PATCH:
- Add this requirements.txt at the ROOT of the GitHub repository, beside the
  `Document Q&A Assistant with RAG & Vector Embedding/` folder.
- Do not paste the folder name into requirements.txt.
- In Streamlit Cloud App Settings / Advanced settings, make sure the Python
  dependencies file is `requirements.txt` (root), not the app folder name.
- Keep Main file path pointing to:
  `Document Q&A Assistant with RAG & Vector Embedding/app.py`

IMPORTANT:
The deployment log shows Python 3.14.10. After fixing the dependency-file parsing,
some packages in this project (especially torch, sentence-transformers, faiss-cpu)
may not yet have compatible wheels for Python 3.14. If the next log reports package
wheel/build or version compatibility errors, change the app's Python version in
Streamlit Cloud advanced settings to Python 3.11 or 3.12 and redeploy.

This patch changes dependency-file placement only. It does not modify app.py,
the RAG pipeline, vector indexes, or Groq settings.
