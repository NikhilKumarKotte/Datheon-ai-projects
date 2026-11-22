# Datheon Testing AI

A polished Streamlit chatbot project with:
- Dark, Light, and Midnight themes
- Streamlit chat interface and progressive token streaming
- Gemini, OpenAI, and Groq provider options plus Demo Mode
- Local SQLite chat history with search, rename, delete, clear-current, and delete-all controls
- Recent-context memory: only the latest 12 messages are included in model context
- Lightweight sentiment cues and explicit answer-style preference detection
- API-key entry in the sidebar or optional `.env` file

## Run on Windows
1. Extract the ZIP and open the folder in VS Code or a terminal.
2. Run `run_windows.bat`, or manually:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

3. Open `http://localhost:8501`.
4. Choose a provider and enter its API key in the sidebar. Demo Mode runs without a key but uses a sample response, not a live LLM.

## API key configuration
You can paste the key into the password field in the app. Alternatively copy `.env.example` to `.env` and set one:
- `GOOGLE_API_KEY=...`
- `OPENAI_API_KEY=...`
- `GROQ_API_KEY=...`

Do not commit or share `.env`. Keys entered in the app are not written to the SQLite chat database.

## Memory design
All chat transcripts remain in local SQLite so you can search and manage history. For model context, the app sends only the latest 12 messages from the active conversation. It derives a small, temporary style profile from recent user wording (for example, requests for detailed answers, concise answers, simple language, or examples) and a basic positive/negative/neutral text cue. This is a lightweight heuristic, not a validated sentiment model or psychological assessment. The profile is used only to gently adjust response presentation; it does not override accuracy or safety. "Reset recent memory" starts a new conversation while preserving old chat history.

## Notes
- Python 3.10–3.12 is recommended.
- Live AI responses require a valid API key and internet access.
- Local chat history is stored in `data/chat_history.db`.
- This is a local student project starter. Add authentication, rate limiting, secure deployment secrets, and deployment hardening before exposing it publicly.


## Semantic memory (new)
Datheon also supports optional local semantic retrieval:
- Every new user and assistant message is indexed in a persistent ChromaDB collection.
- Sentence Transformers (`all-MiniLM-L6-v2`) creates embeddings locally on the machine.
- When a new question arrives, the app retrieves up to 4 relevant older messages from the same conversation.
- The latest 12 messages remain the normal recent context; retrieved older memories supplement them.
- Conversations are isolated by conversation ID. Older messages from other chats are not retrieved.
- A similarity threshold filters weak matches.
- The first run downloads the embedding model and can take longer; allow network access and disk space.
- Conversation text is embedded locally; it is not sent to a third-party embedding API.
- Use the **Use semantic memory** toggle to enable or disable retrieval.

The sentiment/style profile remains a lightweight heuristic based on recent wording. It is used only to adapt answer presentation and is not a psychological assessment.


## Troubleshooting: user message appears but no answer
The current user message is passed into the LangChain message list before streaming. Provider/API-key errors remain visible instead of being cleared by an immediate page rerun. If you see a missing-key message, open **AI configuration** in the sidebar, select your provider, paste its API key, and retry. You can select Demo Mode to verify the chat interface without an API key; Demo Mode is not a live AI model.


## Groq model selection
If Groq returns `model_not_found` / HTTP 404, the selected model ID may have been retired or may not be enabled for your account. Select a currently available model from the model dropdown. The app includes Groq options for `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-120b`, and `openai/gpt-oss-20b`; availability can vary by provider account and model lifecycle.


For Groq, this build offers the current replacement options `openai/gpt-oss-120b` and `openai/gpt-oss-20b`; older Llama model IDs have been removed from the dropdown to avoid selecting the retired IDs by mistake.
