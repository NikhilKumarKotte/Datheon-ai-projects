# AI Provider Setup

The application supports the following providers from the Streamlit sidebar:

| Provider | Provider ID | Default model | API base URL / integration |
|---|---|---|---|
| xAI Grok Cloud | `xai` | `grok-4.7` | `https://api.x.ai/v1` |
| OpenAI | `openai` | `gpt-4o-mini` | Official OpenAI endpoint |
| Google Gemini | `google` | `gemini-2.5-flash` | LangChain Google Generative AI |
| Anthropic Claude | `anthropic` | `claude-3-7-sonnet-latest` | LangChain Anthropic |
| Groq Cloud | `groq` | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` |
| OpenRouter | `openrouter` | `openai/gpt-4o-mini` | `https://openrouter.ai/api/v1` |
| Ollama | `ollama` | `llama3.2` | Local Ollama server |
| Custom compatible API | `custom` | User supplied | User supplied OpenAI-compatible URL |
| Mock | `mock` | Simulated | No external API |

## Configure a provider in `.env`

Copy `.env.example` to `.env`, then set `LLM_PROVIDER`, `LLM_MODEL`, and the matching provider key. Only the selected provider needs a key.

For Grok:

```env
LLM_PROVIDER=xai
LLM_MODEL=grok-4.7
XAI_API_KEY=your_xai_api_key_here
```

Create and manage xAI API keys in the [xAI Console](https://console.x.ai/). API access and model availability depend on the account and current xAI offering.

## Configure through the user interface

1. Start the app with `streamlit run web_app.py`.
2. Select the preferred provider in **Engine Settings**.
3. Select a model.
4. Paste the matching API key into the masked key field. The app does not write this session-entered key to `.env`.
5. Enable the tools you want the agent to use.
6. Submit a query in the chat panel and watch the tool execution trace.

## Security notes

- Never commit `.env` or paste API keys into source code.
- `.env.example` contains placeholders only.
- If a real key was included in a previously shared archive, revoke it and create a replacement.
- The local file reader blocks hidden credential files and private-key file extensions.
- The custom provider must expose an OpenAI-compatible chat-completions API.
- API calls may incur charges according to the provider's pricing and account settings.
