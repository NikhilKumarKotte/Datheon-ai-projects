# Groq fallback setup

This patch keeps Gemini as the primary provider and adds Groq as an independent
fallback for documents that have been converted to text by the existing loader.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure `.env`

Keep your existing Gemini settings and add:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
ENABLE_GROQ_FALLBACK=true
```

Do not commit or share `.env`.

## 3. Run the application

```bash
uvicorn src.extractor.api.app:app --reload --port 8000
```

## 4. Behavior

- Gemini remains the primary provider.
- The existing Gemini model fallback list is attempted first.
- If all Gemini attempts fail, Groq is attempted for text-based documents.
- Groq output is validated with the same Pydantic schema and passed through
  the existing verification agent.
- Binary multimodal documents are not sent to Groq in this integration; they
  remain on the Gemini path.
- Demo mode is unchanged.

## 5. Verify

Try a text-based resume or a PDF/DOCX whose text is extracted by the existing
document loader. On successful fallback, metadata should contain a
`model_used` value beginning with `groq:` and `fallback_used: true`.

Groq is subject to its own model availability, rate limits, and token limits.
