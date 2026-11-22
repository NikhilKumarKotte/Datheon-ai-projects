DARK THEME UI PATCH — Structured LLM Prompt Pipeline & JSON Extractor

This ZIP contains ONLY the updated dashboard HTML. It does not contain Python
source, requirements.txt, virtual environments, or dependencies.

HOW TO APPLY
1. Extract this ZIP.
2. Copy the file:
   src/extractor/api/static/index.html
3. Paste it into the same path inside your existing project, replacing the old
   index.html:
   <your-project>/src/extractor/api/static/index.html
4. Restart the FastAPI server and hard-refresh the browser (Ctrl + F5).

No pip install is required. No dependencies are changed.

The JavaScript, element IDs, API routes, provider selection, upload behavior,
prompt pipeline, and backend are preserved. Only the embedded dashboard CSS
theme is changed to a dark graphite / blue-accent palette.
