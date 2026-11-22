# 📦 Datheon Requirements Directory

This directory provides modular and consolidated dependency manifests for the **Datheon AI Engineering Suite**.

---

## 📁 Directory Structure

| File | Purpose | Corresponding Project |
| :--- | :--- | :--- |
| [`all.txt`](./all.txt) | **Master Unified Suite** | Installs all dependencies for all 4 projects simultaneously |
| [`datheon_testing_ai.txt`](./datheon_testing_ai.txt) | Conversational AI Workspace | `Datheon_Testing_AI` |
| [`document_qa_rag.txt`](./document_qa_rag.txt) | Enterprise RAG & Vector Embeddings | `Document Q-A Assistant with RAG  Vector Embedding` |
| [`multi_tool_agent.txt`](./multi_tool_agent.txt) | Autonomous ReAct Research Agent | `Multi-Tool Autonomous AI Agent` |
| [`structured_llm_extractor.txt`](./structured_llm_extractor.txt) | Document Extraction REST API | `Structured LLM Prompt Pipeline - JSON Extractor` |

---

## 🚀 How to Install

### 1. Install Everything for the Entire Monorepo
From the repository root:
```bash
pip install -r requirements.txt
# Or explicitly:
pip install -r requirements/all.txt
```

### 2. Install Dependencies for a Single Project Only
If you only want to run or test one specific project in an isolated virtual environment:

```bash
# Project 01: Datheon Testing AI
pip install -r requirements/datheon_testing_ai.txt

# Project 02: Document Q-A Assistant (RAG)
pip install -r requirements/document_qa_rag.txt

# Project 03: Multi-Tool Autonomous AI Agent
pip install -r requirements/multi_tool_agent.txt

# Project 04: Structured LLM Extractor
pip install -r requirements/structured_llm_extractor.txt
```

---

## ☁️ Deployment Reference

- **Streamlit Community Cloud**: Automatically reads root `requirements.txt` or the subproject's own `requirements.txt`.
- **Render.com / Docker**: Point build commands to `requirements.txt` in the root or in the subfolder.
