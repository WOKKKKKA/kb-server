---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 56137a95a2d193780765819d098cfe37_2fef3a519ab511f19467525400287e28
    ReservedCode1: PrCQaWi095q9FF/rRpdX5YvtQZtaAGNX6gjVKsQUEWNQxcXdXENcn5Ydx6vPRG23M3Zr8Fm6m1s4x9lnO3hFVJB9o0oPNjbxWfDlUzB2ZqsbhcZn9Aoea6cmVuRwXmFRfqnLmT78zva3twn2whYmja4/quzNrZnOTRGa0b+xmsA5/U2TjfkUmXIsJyo=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 56137a95a2d193780765819d098cfe37_2fef3a519ab511f19467525400287e28
    ReservedCode2: PrCQaWi095q9FF/rRpdX5YvtQZtaAGNX6gjVKsQUEWNQxcXdXENcn5Ydx6vPRG23M3Zr8Fm6m1s4x9lnO3hFVJB9o0oPNjbxWfDlUzB2ZqsbhcZn9Aoea6cmVuRwXmFRfqnLmT78zva3twn2whYmja4/quzNrZnOTRGa0b+xmsA5/U2TjfkUmXIsJyo=
---

# KB Server — Single-File Private Knowledge Base

> **One `.py` file = a complete private knowledge base.** Backend + frontend + hybrid retrieval + OCR, all bundled in a single file. Up and running on your NAS / home server in 5 minutes — your data never leaves your network.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)

---

## Why KB Server

The local knowledge base space is crowded, but most projects are heavy: vector databases, frontend frameworks, multi-container orchestration. **KB Server flips that around:**

| | KB Server | Mainstream (RAGFlow / Dify / AnythingLLM) |
|---|---|---|
| Deployment | **Single file**, one `.py` runs everything | Multi-container, needs vector DB / middleware |
| Vector search | **In-memory numpy matrix**, zero external deps | Requires Milvus / Chroma / ES etc. |
| Frontend | **Embedded Web UI**, nothing to deploy | Separate frontend project |
| Resource usage | Low, runs on small NAS memory | Usually 16GB+ |
| Chinese support | Natively optimized (translation fallback, query quality detection) | Partial |

**One-liner: a "download-and-run, no-fuss" private knowledge base for NAS users and small teams.**

---

## Features

### Retrieval
- **Hybrid search**: vector semantic retrieval + BM25 keyword retrieval, fused with RRF scoring
- **Chinese-optimized**: automatic translation fallback, vague-query detection, related-question suggestions
- **Query cache**: in-memory TTL cache, repeated questions answered instantly

### Documents
- **12 formats**: `txt / md / html / pdf / docx / xlsx / csv / pptx / json` + images
- **Built-in OCR**: scanned PDFs / images auto-transcribed (PaddleOCR, CPU-friendly)
- **Versioning**: same-name docs keep versions, with diff comparison
- **Doc / spec comparison**: side-by-side diff and parameter comparison

### Models
- **Multi-model channels**: separate models for Q&A / chat / compare
- **GPU acceleration**: OpenAI-compatible API (faster_llm / vLLM), 35B models with sub-second responses
- **SSE streaming**: typewriter effect, low time-to-first-token

### Security & Collaboration
- **JWT auth** + multi-user registration with approval
- **Department isolation**: docs scoped by department / role
- **Knowledge graph**: document relationship visualization
- **Stats dashboard**: retrieval volume, doc popularity at a glance

---

## Quick Start

### Docker (recommended)

The repo ships with `Dockerfile` and `requirements.txt` — build locally:

```yaml
# docker-compose.yml
services:
  kb-server:
    build: .                        # local build (Dockerfile included)
    container_name: kb-server
    ports:
      - "8080:8080"
    volumes:
      - ./docs:/kb_persist/docs      # documents
      - ./data:/kb_persist           # SQLite database
    environment:
      - OLLAMA_HOST=http://ollama:11434   # your Ollama endpoint
      - KB_MODEL_ASK=qwen2.5:7b
      - KB_MODEL_CHAT=qwen2.5:14b
    restart: unless-stopped
```

```bash
docker compose up -d --build
# open http://<your-nas>:8080
```

> First build pulls PaddleOCR deps and takes a while. If you don't need OCR, comment out `paddleocr` / `paddlepaddle` in `requirements.txt` and set `KB_OCR_ENABLED=0` for a much smaller image.

### Bare metal

```bash
pip install -r requirements.txt
python kb_server.py
# open http://localhost:8080
```

### One-command launcher (Linux / NAS / macOS)

```bash
chmod +x start.sh
./start.sh
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              KB Server (single file)        │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Web UI  │  │  FastAPI │  │  SSE      │  │
│  │ (embedded)│ │ REST API │  │ streaming │  │
│  └─────────┘  └────┬─────┘  └───────────┘  │
│                    │                        │
│  ┌─────────────────▼─────────────────────┐  │
│  │  hybrid_search                        │  │
│  │  ┌──────────┐   ┌──────────┐          │  │
│  │  │ vector    │   │  BM25    │          │  │
│  │  │(numpy)    │   │(keyword) │          │  │
│  │  └────┬─────┘   └────┬─────┘          │  │
│  │      RRF score fusion                 │  │
│  └───────┼──────────────┼────────────────┘  │
│          │              │                    │
│  ┌───────▼──────────────▼──────┐  ┌────────┐ │
│  │ parse + chunk + embed       │  │ SQLite │ │
│  │ (12 formats + OCR)          │  │ meta   │ │
│  └──────────────┬──────────────┘  └────────┘ │
└─────────────────┼────────────────────────────┘
                  │
        ┌─────────▼─────────┐
        │  Ollama / OpenAI  │
        │  compatible API   │
        └───────────────────┘
```

---

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `KB_DOCS_DIR` | `/kb_persist/docs` | Document storage directory |
| `KB_DB_PATH` | `/kb_persist/kb.db` | SQLite database path |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `KB_MODEL_ASK` | `qwen2.5:7b` | Main Q&A model |
| `KB_MODEL_CHAT` | `qwen2.5:14b` | Chat model |
| `KB_MODEL_COMPARE` | `qwen2.5:14b` | Compare model |
| `KB_GPU_LLM_URL` | `http://localhost:13306/v1` | GPU channel (OpenAI-compatible) |
| `KB_GPU_LLM_MODEL` | `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` | GPU channel model |
| `KB_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `KB_TOP_K` | `6` | Number of retrieved chunks |
| `KB_CHUNK_MAX` | `800` | Chunk size (chars) |
| `KB_CHUNK_OVERLAP` | `100` | Chunk overlap |
| `KB_MAX_CONTEXT_CHARS` | `3000` | Max context chars in prompt |
| `KB_OCR_ENABLED` | `1` | Enable OCR |
| `KB_CACHE_TTL` | `300` | Query cache TTL (seconds) |
| `KB_JWT_SECRET` | `kb_local_secret_change_me` | JWT secret (**change it!**) |

---

## API Overview

| Method | Path | Description |
|---|---|---|
| POST | `/api/kb/ask` | Knowledge base Q&A (SSE) |
| POST | `/api/chat` | Free chat (SSE) |
| POST | `/api/kb/upload-multipart` | Upload document |
| GET | `/api/docs` | Document list |
| DELETE | `/api/docs/{doc_id}` | Delete document |
| GET | `/api/docs/{doc_id}/versions` | Version list |
| POST | `/api/docs/versions/diff` | Version diff |
| POST | `/api/kb/compare` | Document comparison |
| POST | `/api/spec/compare` | Spec comparison |
| GET | `/api/search` | Full-text search |
| GET | `/api/related-questions` | Related questions |
| GET | `/api/kb/graph` | Knowledge graph |
| GET | `/api/stats` | Stats dashboard |
| POST | `/api/login` / `/api/register` | Login / register |
| GET | `/api/health` | Health check |

---

## Screenshots

> TODO: main UI, streaming Q&A, doc diff, knowledge graph, stats dashboard

---

## Roadmap

- [x] Hybrid retrieval (vector + BM25 + RRF)
- [x] 12-format parsing + OCR
- [x] Multi-model channels + GPU acceleration
- [x] Document versioning / diff
- [x] JWT auth + multi-user approval
- [ ] Multi-language README / docs site
- [ ] One-click installers (Synology / QNAP packages)
- [ ] Optional vector backends (Chroma / FAISS)
- [ ] Webhook document auto-sync

---

## License

MIT
*（内容由AI生成，仅供参考）*
