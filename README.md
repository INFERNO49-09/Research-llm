# Research Assistant

A local-first AI research tool with document RAG, web search, and support for any OpenAI-compatible LLM — runs entirely in the browser with no backend required.

![Research Assistant](https://img.shields.io/badge/React-19-blue?logo=react) ![Vite](https://img.shields.io/badge/Vite-8-purple?logo=vite) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Document RAG** — upload `.txt`, `.md`, `.csv`, `.json`, `.log`, `.xml`, `.yaml` files; chunks are indexed with TF-IDF and the top-K most relevant passages are injected into every prompt
- **Web search** — integrates with a self-hosted [SearXNG](https://github.com/searxng/searxng) instance to pull live web results into context
- **Streaming responses** — token-by-token streaming from any OpenAI-compatible API
- **Dual LLM backend** — switch between a local Ollama instance or any OpenAI-compatible API (OpenAI, NVIDIA NIM, Groq, Together AI, LM Studio, etc.)
- **Context panel** — see exactly which document chunks and web results were used to answer each query, with TF-IDF relevance scores
- **Dark mode** — follows system preference automatically
- **No backend** — everything runs in the browser; documents never leave your machine

---

## Getting Started

### Prerequisites

- Node.js 18+
- npm 9+

### Install & run

```bash
git clone https://github.com/INFERNO49-09/Research-llm.git
cd Research-llm
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## LLM Backends

### Ollama (local)

1. Install [Ollama](https://ollama.com)
2. Start it with CORS open:
   ```bash
   OLLAMA_ORIGINS="*" ollama serve
   ```
3. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
4. In the app, select **Ollama** backend and click **Connect**

### Custom API (OpenAI-compatible)

Set the base URL and API key in the **LLM → Custom API** settings panel. Works with:

| Provider | Base URL |
|---|---|
| OpenAI | `https://api.openai.com` |
| NVIDIA NIM | `https://integrate.api.nvidia.com` |
| Groq | `https://api.groq.com/openai` |
| Together AI | `https://api.together.xyz` |
| LM Studio | `http://localhost:1234` |
| Ollama (OpenAI compat) | `http://localhost:11434` |

> **Note:** NVIDIA and some other providers block direct browser requests due to CORS. The app routes these automatically through a local proxy — see [CORS Proxy](#cors-proxy) below.

---

## Web Search via SearXNG

SearXNG must be running locally with JSON output enabled.

**Quick start with Docker:**

```bash
docker run -d \
  -p 8080:8080 \
  -e SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \
  searxng/searxng
```

Make sure `formats: [html, json]` is set in your SearXNG `settings.yml`, then enter `http://localhost:8080` in the **SearXNG** settings tab and toggle web search on.

---

## CORS Proxy

Some APIs (e.g. NVIDIA NIM) block direct browser-to-API requests. The app proxies these through `/nvidia`:

- **Dev:** Vite dev server proxies `/nvidia/*` → `https://integrate.api.nvidia.com/*`
- **Prod (Vercel):** `vercel.json` rewrites handle the same routing

No extra setup needed — just enter `https://integrate.api.nvidia.com` as the base URL and the app handles the rest.

---

## RAG Pipeline

| Parameter | Value |
|---|---|
| Chunk size | 600 characters |
| Chunk overlap | 120 characters |
| Retrieval | TF-IDF scoring |
| Top-K chunks | 5 per query |
| Web results | 5 per query |
| Max file size | 10 MB |

Document text is chunked on upload and scored against each query at inference time — no embeddings or vector DB required.

---

## Deployment

### Vercel

```bash
npm run build
vercel --prod
```

The included `vercel.json` handles the NVIDIA API proxy rewrite automatically.

### Self-hosted

```bash
npm run build
# serve the dist/ folder with any static file server
npx serve dist
```

---

## Project Structure

```
Research-llm/
├── src/
│   ├── App.jsx        # Entire app — RAG engine, LLM backends, UI
│   ├── index.css      # Design tokens (light + dark)
│   └── main.jsx       # React entry point
├── public/
│   └── favicon.svg
├── index.html         # Tabler Icons CDN loaded here
├── vite.config.js     # Dev proxy for CORS-blocked APIs
└── vercel.json        # Vercel rewrite rules for production proxy
```

---

## Tech Stack

- [React 19](https://react.dev) — UI
- [Vite 8](https://vite.dev) — build tool + dev proxy
- [Tabler Icons](https://tabler.io/icons) — icon set
- TF-IDF — retrieval scoring (zero dependencies, implemented from scratch)
- [SearXNG](https://github.com/searxng/searxng) — optional web search
- [Ollama](https://ollama.com) — optional local LLM runner

---

## License

MIT