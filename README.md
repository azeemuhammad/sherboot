# SherBot — Personal RAG Chatbot

**SherBot** is a Retrieval-Augmented Generation (RAG) chatbot for **Muhammad Sher Khan** from Matta, Swat.

It answers questions about his name, father, date of birth, education (Matric & FSc), and hometown using a curated knowledge base.

---

## Stack

| Component     | Choice                                      |
|---------------|---------------------------------------------|
| Embeddings    | `all-MiniLM-L6-v2` (sentence-transformers)  |
| Vector store  | FAISS (IndexFlatIP, cosine)                 |
| LLM           | Gemini 2.0 Flash (default) / OpenAI-compatible |
| Interface     | Streamlit                                   |
| Knowledge     | 17 personal chunks (JSONL)                  |

---

## Quick start (local)

```bash
cd sherbot
pip install -r requirements.txt
streamlit run app.py
```

Optional API key for natural answers:

```bash
export GEMINI_API_KEY=your-key-here
# or DEEPSEEK_API_KEY / OPENAI_API_KEY
```

Without a key the bot still works in **retrieval-only** mode.

**Gemini defaults:**
- Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
- Model: `gemini-2.0-flash`

---

## Knowledge base

`sher_khan_knowledge.jsonl` covers:

- Full name: Muhammad Sher Khan
- Father: Nawab Ali Khan
- Date of birth: 10 January 2005
- Hometown: Matta, Swat
- Matric: The Swat Grammar School, Sambat
- FSc 2022: Govt. Degree College Mingora, Swat

---

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `app.py`
4. (Optional) Add secret:

```toml
GEMINI_API_KEY = "your-key-here"
```

---

## Features

- Named identity: **SherBot**
- Full RAG pipeline
- Conversation history
- Optional source transparency
- Works with or without LLM API key
- Clean Streamlit UI
