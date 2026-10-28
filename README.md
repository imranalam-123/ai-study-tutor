# AI Study Tutor — Source-Based Learning Assistant

An end-to-end portfolio project that turns your own course documents into a study assistant. It retrieves relevant passages from uploaded PDFs, text files, or Markdown notes; answers with source citations; generates practice quizzes; and tracks quiz progress in the current session.

## Features
- Upload multiple PDF, TXT, and Markdown files.
- Built-in Materials Science demo notes so you can test the app immediately.
- TF-IDF retrieval with source and page/section metadata.
- Source-only extractive answer mode with no LLM key required.
- Optional LLM-powered RAG answers through an OpenAI-compatible chat completions API.
- Generated multiple-choice quizzes when LLM access is configured; a source-cloze fallback works without an API key.
- Session quiz history, study score tracking, and CSV downloads.
- Passage library and downloadable index for inspection.
- Basic tests for chunking, retrieval, cited answers, and quiz generation.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Choose **Built-in demo notes** in the sidebar to start without uploading files. For your own notes, select **Upload my documents** and add PDF, TXT, or Markdown files. Image-only scanned PDFs need OCR first; this app extracts existing text, it does not perform OCR.

## Optional generative AI
The app works without an API key. Without a key, answers are created by extracting relevant sentences and practice quizzes use fill-in-the-blank questions derived from source passages. With an OpenAI-compatible key, it can synthesize answers with source IDs and generate conceptual multiple-choice questions.

For local development, create `.streamlit/secrets.toml` (ignored by Git) containing:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_BASE_URL = "https://api.openai.com/v1"
```

Never commit API keys to GitHub. API usage may incur charges.

## Deploy to Streamlit Community Cloud
1. Push the project files to GitHub.
2. Open https://share.streamlit.io/ and sign in.
3. Choose **Create app** and select this repository and branch `main`.
4. Set main file path to `app.py` and deploy.
5. Optionally configure `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` in the app's Secrets settings.

## How it works
1. Extract text and retain page or section metadata.
2. Split text into overlapping chunks.
3. Fit a TF-IDF vectorizer to the loaded chunks and rank them by cosine similarity to the question.
4. Either extract relevant sentences or pass retrieved passages to an optional LLM, requiring source IDs for factual claims.
5. Show the exact retrieved snippets so learners can verify claims.

This is a lightweight RAG prototype. TF-IDF is lexical retrieval, not dense semantic embedding search.

## Limitations and responsible use
- LLM answers and generated questions may be wrong; verify citations against the original material.
- Source-only answers may not synthesize complex multi-step explanations.
- Uploaded content is processed by the running app; do not upload confidential documents to a public demo.
- Progress is session-only; persistent accounts require authentication and a database.
- Evaluate retrieval with known questions and track hit rate, source correctness, faithfulness, latency, and API cost.

## Tests
```bash
python -m unittest discover -s tests -v
```

## Suggested resume bullet
> Built an AI Study Tutor using retrieval-augmented generation (RAG), document parsing, and source-cited question answering; added quiz generation, session-based progress tracking, and a Streamlit interface for interactive learning.

Only claim metrics after measuring them on an explicit evaluation set.
