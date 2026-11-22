# Runtime LLM provider selection

The dashboard now supports request-time selection:

- **Auto**: uses server-configured Gemini and the configured Groq fallback.
- **Google Gemini**: enter a Gemini API key and optional model ID.
- **GroqCloud**: enter a Groq API key and optional model ID.
- **OpenAI-compatible**: enter an OpenAI or OpenRouter key and model ID. For OpenRouter, use `https://openrouter.ai/api/v1` as the base URL; for OpenAI, leave the base URL blank.

The API key is sent in the HTTPS request body to this application server for that extraction only. It is not stored in browser local storage, cookies, `.env`, or a database, and the form clears it after the request. Server operators can technically see credentials that reach their server; only use this feature on a deployment you trust. Always deploy behind HTTPS, avoid request-body logging, and never share API keys in screenshots or logs.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn src.extractor.api.app:app --reload --port 8000
```

## Important limitations

- Groq and OpenAI-compatible providers process text extracted from PDFs and DOCX files.
- Scanned PDFs and image uploads currently require Gemini's multimodal path.
- The OpenAI-compatible option is intended for OpenAI and OpenRouter-compatible endpoints. The server-side endpoint should be restricted to trusted HTTPS hosts before exposing this app publicly.
- Provider/model availability, pricing, quotas, and JSON-mode support depend on the provider and selected model. A successful API response is still checked against the local Pydantic schema.
- This is a request-scoped key feature, not a multi-tenant account vault. For a public SaaS product, add authentication, rate limiting, abuse controls, HTTPS, and secret-redaction in logs before launch.
