import { useState, useRef, useEffect, useCallback } from "react";

// ── RAG Engine ─────────────────────────────────────────────────────────────────

function tokenize(text) {
  return text.toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/).filter(t => t.length > 2);
}

function chunkText(text, size = 600, overlap = 120) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + size, text.length);
    chunks.push({ text: text.slice(start, end) });
    if (end === text.length) break;
    start += size - overlap;
  }
  return chunks;
}

function tfidfScore(query, chunkTxt, allTexts) {
  const qTokens = tokenize(query);
  const cTokens = tokenize(chunkTxt);
  const freq = {};
  cTokens.forEach(t => (freq[t] = (freq[t] || 0) + 1));
  let score = 0;
  const n = allTexts.length;
  qTokens.forEach(qt => {
    if (freq[qt]) {
      const tf = freq[qt] / cTokens.length;
      const df = allTexts.filter(ct => tokenize(ct).includes(qt)).length;
      const idf = Math.log((n + 1) / (df + 1)) + 1;
      score += tf * idf;
    }
  });
  return score;
}

function retrieveChunks(query, docs, topK = 5) {
  const entries = docs.flatMap(doc =>
    doc.chunks.map(c => ({ text: c.text, docName: doc.name, docId: doc.id }))
  );
  if (!entries.length) return [];
  const allTexts = entries.map(e => e.text);
  return entries
    .map(e => ({ ...e, score: tfidfScore(query, e.text, allTexts) }))
    .filter(e => e.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ── SearXNG ────────────────────────────────────────────────────────────────────

async function searxngSearch(host, query, numResults = 5) {
  const url = `${host.replace(/\/$/, "")}/search?q=${encodeURIComponent(query)}&format=json&language=en`;
  const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error(`SearXNG returned ${res.status}`);
  const data = await res.json();
  return (data.results || []).slice(0, numResults).map(r => ({
    title: r.title || "",
    url: r.url || "",
    snippet: r.content || r.snippet || "",
  }));
}

// ── LLM Backends ──────────────────────────────────────────────────────────────

async function callOllama({ host, model, messages, onToken }) {
  const res = await fetch(`${host.replace(/\/$/, "")}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, stream: true }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Ollama ${res.status}: ${text.slice(0, 200)}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value, { stream: true }).split("\n").filter(Boolean)) {
      try {
        const token = JSON.parse(line)?.message?.content || "";
        if (token) { full += token; onToken(full); }
      } catch (_) { }
    }
  }
  return full;
}

async function callCustomAPI({ baseUrl, apiKey, model, messages, onToken }) {
  const res = await fetch(`${baseUrl.replace(/\/$/, "")}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(apiKey && { Authorization: `Bearer ${apiKey}` }),
    },
    body: JSON.stringify({ model, messages, stream: true }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text.slice(0, 200)}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value, { stream: true }).split("\n")) {
      const trimmed = line.replace(/^data: /, "").trim();
      if (!trimmed || trimmed === "[DONE]") continue;
      try {
        const token = JSON.parse(trimmed)?.choices?.[0]?.delta?.content || "";
        if (token) { full += token; onToken(full); }
      } catch (_) { }
    }
  }
  return full;
}

async function fetchOllamaModels(host) {
  const res = await fetch(`${host.replace(/\/$/, "")}/api/tags`, { signal: AbortSignal.timeout(4000) });
  if (!res.ok) throw new Error("unreachable");
  const data = await res.json();
  return (data.models || []).map(m => m.name);
}

// ── Markdown renderer ──────────────────────────────────────────────────────────

function parseLine(line) {
  const parts = [];
  let rem = line, key = 0;
  while (rem) {
    const bm = rem.match(/\*\*(.*?)\*\*/);
    const cm = rem.match(/`([^`]+)`/);
    if (!bm && !cm) { parts.push(rem); break; }
    const bi = bm ? rem.indexOf(bm[0]) : Infinity;
    const ci = cm ? rem.indexOf(cm[0]) : Infinity;
    if (bi < ci) {
      if (bi > 0) parts.push(rem.slice(0, bi));
      parts.push(<strong key={key++} style={{ fontWeight: 500 }}>{bm[1]}</strong>);
      rem = rem.slice(bi + bm[0].length);
    } else {
      if (ci > 0) parts.push(rem.slice(0, ci));
      parts.push(<code key={key++} style={{ fontFamily: "var(--font-mono)", fontSize: 12, background: "var(--color-background-tertiary)", padding: "1px 5px", borderRadius: 4, border: "0.5px solid var(--color-border-tertiary)" }}>{cm[1]}</code>);
      rem = rem.slice(ci + cm[0].length);
    }
  }
  return parts;
}

function renderMD(text) {
  return text.split("\n").map((line, i) => {
    if (line.startsWith("### ")) return <h3 key={i} style={{ margin: "10px 0 3px", fontSize: 13, fontWeight: 500 }}>{line.slice(4)}</h3>;
    if (line.startsWith("## ")) return <h2 key={i} style={{ margin: "13px 0 4px", fontSize: 14, fontWeight: 500 }}>{line.slice(3)}</h2>;
    if (line.startsWith("# ")) return <h1 key={i} style={{ margin: "14px 0 6px", fontSize: 15, fontWeight: 500 }}>{line.slice(2)}</h1>;
    if (line.startsWith("- ") || line.startsWith("* "))
      return <div key={i} style={{ display: "flex", gap: 7, margin: "2px 0", paddingLeft: 6 }}><span style={{ color: "var(--color-text-secondary)", flexShrink: 0 }}>•</span><span style={{ flex: 1 }}>{parseLine(line.slice(2))}</span></div>;
    if (/^\d+\. /.test(line)) {
      const m = line.match(/^(\d+)\. (.*)/);
      return <div key={i} style={{ display: "flex", gap: 7, margin: "2px 0", paddingLeft: 6 }}><span style={{ color: "var(--color-text-secondary)", flexShrink: 0 }}>{m[1]}.</span><span>{parseLine(m[2])}</span></div>;
    }
    if (line.startsWith("> ")) return <div key={i} style={{ borderLeft: "2px solid var(--color-border-secondary)", paddingLeft: 10, margin: "4px 0", color: "var(--color-text-secondary)", fontStyle: "italic", fontSize: 12 }}>{parseLine(line.slice(2))}</div>;
    if (line === "") return <div key={i} style={{ height: 6 }} />;
    return <p key={i} style={{ margin: "2px 0", lineHeight: 1.65 }}>{parseLine(line)}</p>;
  });
}

// ── Status dot ────────────────────────────────────────────────────────────────

function StatusDot({ status }) {
  const colors = { ok: "var(--color-text-success)", error: "var(--color-text-danger)", checking: "var(--color-text-warning)", unknown: "var(--color-border-secondary)" };
  return <span style={{ width: 7, height: 7, borderRadius: "50%", background: colors[status] || colors.unknown, display: "inline-block", flexShrink: 0 }} />;
}

const SUGGESTIONS = [
  "Summarize the uploaded documents",
  "What are the key themes?",
  "Extract main concepts",
  "Find contradictions or gaps",
];

const DEFAULT_OLLAMA_MODELS = ["llama3.2", "llama3.1", "llama3", "mistral", "mistral-nemo", "gemma3", "gemma2", "qwen2.5", "phi4", "deepseek-r1", "codellama", "neural-chat"];

// ── App ────────────────────────────────────────────────────────────────────────

export default function ResearchAssistant() {
  // ── Documents
  const [docs, setDocs] = useState([]);
  const [activeDoc, setActiveDoc] = useState(null);
  const [processingFiles, setProcessingFiles] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // ── Chat
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const [webResults, setWebResults] = useState([]);
  const [streamText, setStreamText] = useState("");

  // ── LLM config
  const [backend, setBackend] = useState("ollama"); // "ollama" | "custom"
  const [ollamaHost, setOllamaHost] = useState("http://localhost:11434");
  const [ollamaModel, setOllamaModel] = useState("llama3.2");
  const [ollamaModels, setOllamaModels] = useState(DEFAULT_OLLAMA_MODELS);
  const [ollamaStatus, setOllamaStatus] = useState("unknown");
  const [customBaseUrl, setCustomBaseUrl] = useState("https://api.openai.com");
  const [customApiKey, setCustomApiKey] = useState("");
  const [customModel, setCustomModel] = useState("gpt-4o-mini");
  const [customStatus, setCustomStatus] = useState("unknown");

  // ── SearXNG config
  const [searxEnabled, setSearxEnabled] = useState(false);
  const [searxHost, setSearxHost] = useState("http://localhost:8080");
  const [searxStatus, setSearxStatus] = useState("unknown");

  // ── UI
  const [activeTab, setActiveTab] = useState("llm"); // "llm" | "searx" | "rag"
  const [showSettings, setShowSettings] = useState(false);
  const [showKeyInput, setShowKeyInput] = useState(false);

  const fileRef = useRef(null);
  const chatRef = useRef(null);
  const textRef = useRef(null);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading, streamText]);

  // ── Ollama connect
  const connectOllama = useCallback(async (host = ollamaHost) => {
    setOllamaStatus("checking");
    try {
      const models = await fetchOllamaModels(host);
      setOllamaModels(models.length ? models : DEFAULT_OLLAMA_MODELS);
      if (models.length && !models.includes(ollamaModel)) setOllamaModel(models[0]);
      setOllamaStatus("ok");
    } catch { setOllamaStatus("error"); }
  }, [ollamaHost, ollamaModel]);

  // ── Custom API test
  const testCustomAPI = useCallback(async () => {
    setCustomStatus("checking");
    try {
      const res = await fetch(`${customBaseUrl.replace(/\/$/, "")}/v1/models`, {
        headers: { ...(customApiKey && { Authorization: `Bearer ${customApiKey}` }) },
        signal: AbortSignal.timeout(5000),
      });
      setCustomStatus(res.ok ? "ok" : "error");
    } catch { setCustomStatus("error"); }
  }, [customBaseUrl, customApiKey]);

  // ── SearXNG test
  const testSearx = useCallback(async (host = searxHost) => {
    setSearxStatus("checking");
    try {
      await searxngSearch(host, "test", 1);
      setSearxStatus("ok");
    } catch { setSearxStatus("error"); }
  }, [searxHost]);

  useEffect(() => { connectOllama(); }, []);

  // ── File processing
  const processFiles = async (files) => {
    setProcessingFiles(true);
    for (const file of Array.from(files)) {
      if (file.size > 10_000_000) continue;
      try {
        const text = await file.text();
        if (!text.trim()) continue;
        setDocs(prev => [...prev, { id: crypto.randomUUID(), name: file.name, size: file.size, text, chunks: chunkText(text), addedAt: Date.now() }]);
      } catch (_) { }
    }
    setProcessingFiles(false);
  };

  // ── Send
  const send = async (overrideInput) => {
    const query = (overrideInput ?? input).trim();
    if (!query || loading) return;

    const userMsg = { role: "user", content: query };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    if (textRef.current) textRef.current.style.height = "auto";
    setLoading(true);
    setStreamText("");
    setSources([]);
    setWebResults([]);

    const retrieved = retrieveChunks(query, docs, 5);
    setSources(retrieved);

    let webContext = "";
    let fetchedWebResults = [];
    if (searxEnabled) {
      try {
        fetchedWebResults = await searxngSearch(searxHost, query, 5);
        setWebResults(fetchedWebResults);
        if (fetchedWebResults.length) {
          webContext = `\n\n<web_search_results>\n${fetchedWebResults.map((r, i) =>
            `[Web ${i + 1}] ${r.title}\nURL: ${r.url}\n${r.snippet}`
          ).join("\n\n")}\n</web_search_results>`;
        }
      } catch (e) {
        webContext = `\n\n<web_search_error>SearXNG unavailable: ${e.message}</web_search_error>`;
      }
    }

    const docContext = retrieved.length > 0
      ? `\n\n<document_context>\n${retrieved.map((r, i) =>
        `[Doc chunk ${i + 1} — "${r.docName}"]:\n${r.text}`
      ).join("\n\n")}\n</document_context>`
      : "";

    const system = `You are a precise AI Research Assistant.${docContext}${webContext}

Guidelines:
- Cite document chunks as [Doc chunk N from "filename"] and web results as [Web N]
- Use markdown: ## headers, - bullets, **bold key terms**, \`inline code\`
- Clearly distinguish document-sourced facts, web-sourced facts, and general knowledge
- Be thorough but focused. State uncertainty explicitly`;

    const llmMessages = [
      { role: "system", content: system },
      ...history.map(m => ({ role: m.role, content: m.content })),
    ];

    try {
      let reply = "";
      const onToken = (partial) => setStreamText(partial);

      if (backend === "ollama") {
        reply = await callOllama({ host: ollamaHost, model: ollamaModel, messages: llmMessages, onToken });
      } else {
        reply = await callCustomAPI({ baseUrl: customBaseUrl, apiKey: customApiKey, model: customModel, messages: llmMessages, onToken });
      }

      setStreamText("");
      setMessages(prev => [...prev, { role: "assistant", content: reply || "(empty response)", sources: retrieved, webResults: fetchedWebResults }]);
    } catch (err) {
      setStreamText("");
      const hint = backend === "ollama"
        ? "\n\n> Start Ollama with: `OLLAMA_ORIGINS=\"*\" ollama serve`"
        : "\n\n> Check your base URL and API key in settings.";
      setMessages(prev => [...prev, { role: "assistant", content: `**Error:** ${err.message}${hint}`, sources: [], webResults: [] }]);
    } finally {
      setLoading(false);
    }
  };

  const totalChunks = docs.reduce((a, d) => a + d.chunks.length, 0);
  const totalChars = docs.reduce((a, d) => a + d.text.length, 0);
  const activeModel = backend === "ollama" ? ollamaModel : customModel;
  const activeStatus = backend === "ollama" ? ollamaStatus : customStatus;

  const tabStyle = (t) => ({
    fontSize: 11, padding: "4px 10px", border: "none", background: "none", cursor: "pointer",
    color: activeTab === t ? "var(--color-text-primary)" : "var(--color-text-secondary)",
    fontFamily: "var(--font-sans)", fontWeight: activeTab === t ? 500 : 400,
    borderBottom: `2px solid ${activeTab === t ? "var(--color-text-warning)" : "transparent"}`,
    transition: "all 0.15s",
  });

  // ── Render
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", fontFamily: "var(--font-sans)", fontSize: 14, color: "var(--color-text-primary)", background: "var(--color-background-tertiary)" }}>

      {/* Header */}
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "9px 14px", background: "var(--color-background-primary)", borderBottom: "0.5px solid var(--color-border-tertiary)", flexShrink: 0, gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <div style={{ width: 30, height: 30, borderRadius: "var(--border-radius-md)", background: "var(--color-background-warning)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            <i className="ti ti-brain" style={{ fontSize: 16, color: "var(--color-text-warning)" }} aria-hidden="true" />
          </div>
          <div>
            <div style={{ fontWeight: 500, fontSize: 14, lineHeight: 1.2 }}>Research Assistant</div>
            <div style={{ fontSize: 11, color: "var(--color-text-secondary)", lineHeight: 1.2, display: "flex", alignItems: "center", gap: 5 }}>
              <StatusDot status={activeStatus} />
              {activeModel}
              {searxEnabled && <><span style={{ opacity: 0.4 }}>·</span><StatusDot status={searxStatus} /><span>SearXNG</span></>}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{ fontSize: 11, color: "var(--color-text-secondary)", padding: "3px 9px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-tertiary)", display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: docs.length > 0 ? "var(--color-text-success)" : "var(--color-border-secondary)", display: "inline-block" }} />
            {docs.length > 0 ? `${docs.length} doc${docs.length > 1 ? "s" : ""} · ${totalChunks} chunks` : "No documents"}
          </div>
          <button onClick={() => setSearxEnabled(s => !s)}
            style={{ fontSize: 11, padding: "3px 9px", borderRadius: "var(--border-radius-md)", border: `0.5px solid ${searxEnabled ? "var(--color-border-success)" : "var(--color-border-tertiary)"}`, background: searxEnabled ? "var(--color-background-success)" : "var(--color-background-secondary)", color: searxEnabled ? "var(--color-text-success)" : "var(--color-text-secondary)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
            <i className="ti ti-search" style={{ fontSize: 13 }} aria-hidden="true" />
            SearXNG {searxEnabled ? "on" : "off"}
          </button>
          <button onClick={() => setShowSettings(s => !s)}
            style={{ fontSize: 11, padding: "3px 9px", borderRadius: "var(--border-radius-md)", border: `0.5px solid ${showSettings ? "var(--color-border-secondary)" : "var(--color-border-tertiary)"}`, background: showSettings ? "var(--color-background-secondary)" : "transparent", color: "var(--color-text-secondary)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
            <i className="ti ti-settings" style={{ fontSize: 13 }} aria-hidden="true" />
          </button>
          {messages.length > 0 && (
            <button onClick={() => { setMessages([]); setSources([]); setWebResults([]); setStreamText(""); }}
              style={{ fontSize: 11, padding: "3px 9px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-tertiary)", background: "transparent", color: "var(--color-text-secondary)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
              <i className="ti ti-refresh" style={{ fontSize: 13 }} aria-hidden="true" />
              New chat
            </button>
          )}
        </div>
      </header>

      {/* Settings panel */}
      {showSettings && (
        <div style={{ background: "var(--color-background-primary)", borderBottom: "0.5px solid var(--color-border-tertiary)", flexShrink: 0 }}>
          {/* Tabs */}
          <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)", padding: "0 14px" }}>
            {[["llm", "ti-cpu", "LLM"], ["searx", "ti-search", "SearXNG"], ["rag", "ti-puzzle", "RAG"]].map(([t, icon, label]) => (
              <button key={t} onClick={() => setActiveTab(t)} style={tabStyle(t)}>
                <i className={`ti ${icon}`} style={{ fontSize: 12, marginRight: 4 }} aria-hidden="true" />{label}
              </button>
            ))}
          </div>

          <div style={{ padding: "14px 16px" }}>

            {/* LLM tab */}
            {activeTab === "llm" && (
              <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                {/* Backend switch */}
                <div style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 180 }}>
                  <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Backend</div>
                  <div style={{ display: "flex", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-tertiary)", overflow: "hidden", width: "fit-content" }}>
                    {[["ollama", "Ollama"], ["custom", "Custom API"]].map(([b, label]) => (
                      <button key={b} onClick={() => setBackend(b)}
                        style={{ fontSize: 11, padding: "5px 12px", border: "none", background: backend === b ? "var(--color-background-warning)" : "var(--color-background-secondary)", color: backend === b ? "var(--color-text-warning)" : "var(--color-text-secondary)", cursor: "pointer", fontFamily: "var(--font-sans)", fontWeight: backend === b ? 500 : 400 }}>
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Ollama config */}
                {backend === "ollama" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, minWidth: 280 }}>
                    <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: 6 }}>
                      <StatusDot status={ollamaStatus} /> Ollama
                    </div>
                    <div style={{ display: "flex", gap: 6 }}>
                      <input value={ollamaHost} onChange={e => setOllamaHost(e.target.value)} placeholder="http://localhost:11434"
                        style={{ flex: 1, fontSize: 12, padding: "5px 9px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", outline: "none" }} />
                      <button onClick={() => connectOllama(ollamaHost)}
                        style={{ fontSize: 11, padding: "5px 11px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", cursor: "pointer" }}>
                        Connect
                      </button>
                    </div>
                    <select value={ollamaModel} onChange={e => setOllamaModel(e.target.value)}
                      style={{ fontSize: 12, padding: "5px 9px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", fontFamily: "var(--font-sans)" }}>
                      {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                    {ollamaStatus === "error" && (
                      <div style={{ fontSize: 11, color: "var(--color-text-secondary)", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "8px 10px", lineHeight: 1.7, border: "0.5px solid var(--color-border-tertiary)" }}>
                        Start Ollama with CORS open:<br />
                        <code style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-warning)", fontSize: 11 }}>OLLAMA_ORIGINS="*" ollama serve</code>
                      </div>
                    )}
                  </div>
                )}

                {/* Custom API config */}
                {backend === "custom" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, minWidth: 300 }}>
                    <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: 6 }}>
                      <StatusDot status={customStatus} /> OpenAI-compatible API
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                      <input value={customBaseUrl} onChange={e => setCustomBaseUrl(e.target.value)} placeholder="https://api.openai.com"
                        style={{ fontSize: 12, padding: "5px 9px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", outline: "none" }} />
                      <input value={customModel} onChange={e => setCustomModel(e.target.value)} placeholder="gpt-4o-mini"
                        style={{ fontSize: 12, padding: "5px 9px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", outline: "none" }} />
                    </div>
                    <div style={{ display: "flex", gap: 6 }}>
                      <div style={{ flex: 1, position: "relative" }}>
                        <input
                          type={showKeyInput ? "text" : "password"}
                          value={customApiKey}
                          onChange={e => setCustomApiKey(e.target.value)}
                          placeholder="sk-… (leave blank if not required)"
                          style={{ width: "100%", fontSize: 12, padding: "5px 30px 5px 9px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", outline: "none", boxSizing: "border-box" }}
                        />
                        <button onClick={() => setShowKeyInput(s => !s)}
                          style={{ position: "absolute", right: 6, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--color-text-secondary)", padding: 0, display: "flex" }}>
                          <i className={`ti ${showKeyInput ? "ti-eye-off" : "ti-eye"}`} style={{ fontSize: 14 }} />
                        </button>
                      </div>
                      <button onClick={testCustomAPI}
                        style={{ fontSize: 11, padding: "5px 11px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", cursor: "pointer", whiteSpace: "nowrap" }}>
                        Test
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>
                      Compatible with OpenAI, Groq, Together AI, LM Studio, Ollama (<code style={{ fontFamily: "var(--font-mono)" }}>/v1</code>), and any OpenAI-compatible endpoint.
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* SearXNG tab */}
            {activeTab === "searx" && (
              <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, minWidth: 280 }}>
                  <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: 6 }}>
                    <StatusDot status={searxStatus} /> SearXNG instance
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    <input value={searxHost} onChange={e => setSearxHost(e.target.value)} placeholder="http://localhost:8080"
                      style={{ flex: 1, fontSize: 12, padding: "5px 9px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", outline: "none" }} />
                    <button onClick={() => testSearx(searxHost)}
                      style={{ fontSize: 11, padding: "5px 11px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", cursor: "pointer" }}>
                      Test
                    </button>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 12 }}>
                      <input type="checkbox" checked={searxEnabled} onChange={e => setSearxEnabled(e.target.checked)} />
                      Enable web search via SearXNG
                    </label>
                  </div>
                  {searxStatus === "error" && (
                    <div style={{ fontSize: 11, color: "var(--color-text-secondary)", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "8px 10px", lineHeight: 1.7, border: "0.5px solid var(--color-border-tertiary)" }}>
                      Start SearXNG with JSON API enabled. Ensure <code style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>formats: [html, json]</code> is in your settings and CORS headers are set.
                    </div>
                  )}
                </div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", lineHeight: 1.8, minWidth: 200, maxWidth: 280 }}>
                  <div style={{ fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 4 }}>Quick setup</div>
                  <code style={{ fontFamily: "var(--font-mono)", fontSize: 10, display: "block", background: "var(--color-background-secondary)", padding: "8px 10px", borderRadius: "var(--border-radius-md)", lineHeight: 1.9 }}>
                    docker run -d \<br />
                    {"  "}-p 8080:8080 \<br />
                    {"  "}searxng/searxng
                  </code>
                </div>
              </div>
            )}

            {/* RAG tab */}
            {activeTab === "rag" && (
              <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
                {[
                  { label: "Chunk size", val: "600 chars", desc: "Max chars per chunk" },
                  { label: "Overlap", val: "120 chars", desc: "Overlap between chunks" },
                  { label: "Top-K retrieval", val: "5 chunks", desc: "Chunks sent as context" },
                  { label: "Scoring", val: "TF-IDF", desc: "Term frequency–inverse document frequency" },
                  { label: "Web results", val: "5 results", desc: "SearXNG results per query" },
                ].map(({ label, val, desc }) => (
                  <div key={label} style={{ minWidth: 140 }}>
                    <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>{val}</div>
                    <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 1 }}>{desc}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Body */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0 }}>

        {/* Left: Documents */}
        <div style={{ width: 210, background: "var(--color-background-primary)", borderRight: "0.5px solid var(--color-border-tertiary)", display: "flex", flexDirection: "column", flexShrink: 0 }}>
          <div style={{ padding: "10px 10px 8px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 8 }}>
              <i className="ti ti-files" style={{ fontSize: 12, marginRight: 5 }} aria-hidden="true" />Documents
            </div>
            <div
              onDrop={e => { e.preventDefault(); setDragOver(false); processFiles(e.dataTransfer.files); }}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onClick={() => fileRef.current?.click()}
              style={{ border: `1.5px dashed ${dragOver ? "var(--color-border-info)" : "var(--color-border-secondary)"}`, borderRadius: "var(--border-radius-md)", padding: "12px 8px", textAlign: "center", cursor: "pointer", background: dragOver ? "var(--color-background-info)" : "var(--color-background-secondary)", transition: "all 0.15s" }}>
              {processingFiles
                ? <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}><i className="ti ti-loader-2" style={{ fontSize: 16, display: "block", marginBottom: 3 }} aria-hidden="true" />Processing…</div>
                : <>
                  <i className="ti ti-upload" style={{ fontSize: 17, color: dragOver ? "var(--color-text-info)" : "var(--color-text-secondary)", display: "block", marginBottom: 3 }} aria-hidden="true" />
                  <div style={{ fontSize: 11, color: "var(--color-text-secondary)", lineHeight: 1.4 }}>Drop or click to upload</div>
                  <div style={{ fontSize: 10, color: "var(--color-text-secondary)", opacity: 0.6, marginTop: 2 }}>.txt .md .csv .json .log .xml</div>
                </>
              }
            </div>
            <input ref={fileRef} type="file" multiple accept=".txt,.md,.csv,.json,.log,.xml,.yaml,.yml" onChange={e => processFiles(e.target.files)} style={{ display: "none" }} />
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "5px 6px" }}>
            {docs.length === 0
              ? <div style={{ padding: "18px 8px", textAlign: "center", color: "var(--color-text-secondary)", fontSize: 11, lineHeight: 1.7 }}><i className="ti ti-database-off" style={{ fontSize: 20, display: "block", marginBottom: 5, opacity: 0.4 }} aria-hidden="true" />Upload files to enable RAG retrieval.</div>
              : docs.map(doc => (
                <div key={doc.id} onClick={() => setActiveDoc(activeDoc?.id === doc.id ? null : doc)}
                  style={{ padding: "7px 8px", borderRadius: "var(--border-radius-md)", marginBottom: 2, border: `0.5px solid ${activeDoc?.id === doc.id ? "var(--color-border-info)" : "var(--color-border-tertiary)"}`, background: activeDoc?.id === doc.id ? "var(--color-background-info)" : "var(--color-background-secondary)", cursor: "pointer", display: "flex", alignItems: "flex-start", gap: 7 }}>
                  <i className="ti ti-file-text" style={{ fontSize: 13, color: "var(--color-text-secondary)", flexShrink: 0, marginTop: 1 }} aria-hidden="true" />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{doc.name}</div>
                    <div style={{ fontSize: 10, color: "var(--color-text-secondary)" }}>{doc.chunks.length} chunks · {(doc.size / 1000).toFixed(1)}kb</div>
                  </div>
                  <button onClick={e => { e.stopPropagation(); setDocs(d => d.filter(x => x.id !== doc.id)); if (activeDoc?.id === doc.id) setActiveDoc(null); }}
                    style={{ background: "none", border: "none", color: "var(--color-text-secondary)", cursor: "pointer", padding: 2, lineHeight: 1, flexShrink: 0 }} aria-label={`Remove ${doc.name}`}>
                    <i className="ti ti-x" style={{ fontSize: 12 }} />
                  </button>
                </div>
              ))
            }
          </div>

          {activeDoc && (
            <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: "8px 10px", maxHeight: 150, overflowY: "auto", flexShrink: 0 }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 5 }}>
                <i className="ti ti-eye" style={{ fontSize: 11, marginRight: 4 }} aria-hidden="true" />Preview
              </div>
              <pre style={{ fontSize: 10, color: "var(--color-text-secondary)", lineHeight: 1.5, fontFamily: "var(--font-mono)", whiteSpace: "pre-wrap", wordBreak: "break-all", margin: 0 }}>
                {activeDoc.text.slice(0, 500)}{activeDoc.text.length > 500 ? "…" : ""}
              </pre>
            </div>
          )}
        </div>

        {/* Center: Chat */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
          <div ref={chatRef} style={{ flex: 1, overflowY: "auto", padding: "16px 18px", display: "flex", flexDirection: "column", gap: 14 }}>

            {messages.length === 0 && (
              <div style={{ textAlign: "center", padding: "44px 20px" }}>
                <div style={{ width: 52, height: 52, borderRadius: "var(--border-radius-lg)", background: "var(--color-background-warning)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 14px" }}>
                  <i className="ti ti-brain" style={{ fontSize: 26, color: "var(--color-text-warning)" }} aria-hidden="true" />
                </div>
                <div style={{ fontWeight: 500, fontSize: 15, marginBottom: 6 }}>Research Assistant</div>
                <div style={{ fontSize: 12, lineHeight: 1.7, maxWidth: 380, margin: "0 auto 6px", color: "var(--color-text-secondary)" }}>
                  <strong style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>{backend === "ollama" ? `Ollama · ${ollamaModel}` : `Custom API · ${customModel}`}</strong>
                  {searxEnabled && <span style={{ color: "var(--color-text-success)" }}> · SearXNG on</span>}
                  {docs.length > 0 && <span style={{ color: "var(--color-text-success)" }}> · {totalChunks} chunks</span>}
                </div>
                <div style={{ fontSize: 12, lineHeight: 1.7, maxWidth: 380, margin: "0 auto 20px", color: "var(--color-text-secondary)" }}>
                  Upload documents for RAG, enable SearXNG for live web results, or ask anything directly.
                </div>
                <div style={{ display: "flex", gap: 7, justifyContent: "center", flexWrap: "wrap", maxWidth: 420, margin: "0 auto" }}>
                  {SUGGESTIONS.map(s => (
                    <button key={s} onClick={() => send(s)}
                      style={{ fontSize: 11, padding: "5px 11px", borderRadius: 20, border: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", cursor: "pointer" }}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} style={{ display: "flex", gap: 9, alignItems: "flex-start", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
                {msg.role === "assistant" && (
                  <div style={{ width: 27, height: 27, borderRadius: "50%", background: "var(--color-background-warning)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 2 }}>
                    <i className="ti ti-brain" style={{ fontSize: 14, color: "var(--color-text-warning)" }} aria-hidden="true" />
                  </div>
                )}
                <div style={{ maxWidth: msg.role === "user" ? "65%" : "78%", background: msg.role === "user" ? "var(--color-background-info)" : "var(--color-background-primary)", border: `0.5px solid ${msg.role === "user" ? "var(--color-border-info)" : "var(--color-border-tertiary)"}`, borderRadius: msg.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px", padding: "9px 12px", fontSize: 13 }}>
                  {msg.role === "user" ? <span>{msg.content}</span> : <div>{renderMD(msg.content)}</div>}

                  {(msg.sources?.length > 0 || msg.webResults?.length > 0) && (
                    <div style={{ marginTop: 9, paddingTop: 8, borderTop: "0.5px solid var(--color-border-tertiary)", display: "flex", flexDirection: "column", gap: 6 }}>
                      {msg.sources?.length > 0 && (
                        <div>
                          <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4, display: "flex", alignItems: "center", gap: 4 }}>
                            <i className="ti ti-database" style={{ fontSize: 11 }} aria-hidden="true" />Doc chunks · {msg.sources.length}
                          </div>
                          {msg.sources.slice(0, 3).map((s, si) => (
                            <div key={si} style={{ fontSize: 10, color: "var(--color-text-secondary)", marginBottom: 2, display: "flex", gap: 5 }}>
                              <span style={{ color: "var(--color-text-warning)", fontWeight: 500, flexShrink: 0 }}>[{si + 1}]</span>
                              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.docName}</span>
                              <span style={{ opacity: 0.5, flexShrink: 0 }}>·{s.score.toFixed(3)}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {msg.webResults?.length > 0 && (
                        <div>
                          <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4, display: "flex", alignItems: "center", gap: 4 }}>
                            <i className="ti ti-search" style={{ fontSize: 11 }} aria-hidden="true" />Web results · {msg.webResults.length}
                          </div>
                          {msg.webResults.slice(0, 3).map((r, ri) => (
                            <div key={ri} style={{ fontSize: 10, color: "var(--color-text-secondary)", marginBottom: 2, display: "flex", gap: 5 }}>
                              <span style={{ color: "var(--color-text-success)", fontWeight: 500, flexShrink: 0 }}>[W{ri + 1}]</span>
                              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={r.url}>{r.title || r.url}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Streaming */}
            {loading && streamText && (
              <div style={{ display: "flex", gap: 9, alignItems: "flex-start" }}>
                <div style={{ width: 27, height: 27, borderRadius: "50%", background: "var(--color-background-warning)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  <i className="ti ti-brain" style={{ fontSize: 14, color: "var(--color-text-warning)" }} aria-hidden="true" />
                </div>
                <div style={{ maxWidth: "78%", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "14px 14px 14px 4px", padding: "9px 12px", fontSize: 13 }}>
                  <div>{renderMD(streamText)}</div>
                  <span style={{ display: "inline-block", width: 2, height: 13, background: "var(--color-text-warning)", marginLeft: 2, verticalAlign: "text-bottom", animation: "blink 1s step-end infinite" }} />
                  <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}`}</style>
                </div>
              </div>
            )}

            {loading && !streamText && (
              <div style={{ display: "flex", gap: 9, alignItems: "flex-start" }}>
                <div style={{ width: 27, height: 27, borderRadius: "50%", background: "var(--color-background-warning)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  <i className="ti ti-brain" style={{ fontSize: 14, color: "var(--color-text-warning)" }} aria-hidden="true" />
                </div>
                <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "14px 14px 14px 4px", padding: "11px 14px", display: "flex", gap: 5, alignItems: "center" }}>
                  <style>{`@keyframes dot{0%,80%,100%{transform:scale(0.5);opacity:0.4}40%{transform:scale(1);opacity:1}}`}</style>
                  {[0, 1, 2].map(j => <div key={j} style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--color-text-warning)", animation: `dot 1s ${j * 0.18}s infinite ease-in-out both` }} />)}
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div style={{ padding: "9px 12px", borderTop: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-primary)", flexShrink: 0 }}>
            <div style={{ display: "flex", gap: 7, alignItems: "flex-end", background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: "var(--border-radius-lg)", padding: "5px 5px 5px 12px" }}>
              <textarea ref={textRef} value={input}
                onChange={e => { setInput(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px"; }}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder={docs.length > 0 ? `Ask about ${docs.length} document${docs.length > 1 ? "s" : ""}…` : "Ask anything…"}
                rows={1}
                style={{ flex: 1, background: "none", border: "none", outline: "none", color: "var(--color-text-primary)", fontSize: 13, resize: "none", fontFamily: "var(--font-sans)", lineHeight: 1.6, padding: "4px 0", maxHeight: 120, overflowY: "auto" }}
              />
              <button onClick={() => send()} disabled={loading || !input.trim()} aria-label="Send"
                style={{ width: 32, height: 32, flexShrink: 0, borderRadius: "var(--border-radius-md)", border: "none", background: input.trim() && !loading ? "var(--color-background-warning)" : "var(--color-background-tertiary)", color: input.trim() && !loading ? "var(--color-text-warning)" : "var(--color-text-secondary)", cursor: input.trim() && !loading ? "pointer" : "default", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.15s" }}>
                <i className="ti ti-send" style={{ fontSize: 15 }} />
              </button>
            </div>
            <div style={{ fontSize: 10, color: "var(--color-text-secondary)", marginTop: 5, textAlign: "center", opacity: 0.7 }}>
              {backend === "ollama" ? `Ollama · ${ollamaModel}` : `Custom · ${customModel}`}
              {docs.length > 0 ? ` · ${totalChunks} chunks` : ""}
              {searxEnabled ? " · SearXNG on" : ""}
            </div>
          </div>
        </div>

        {/* Right: Context */}
        <div style={{ width: 208, background: "var(--color-background-primary)", borderLeft: "0.5px solid var(--color-border-tertiary)", display: "flex", flexDirection: "column", flexShrink: 0 }}>
          <div style={{ padding: "10px 10px 8px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.05em", textTransform: "uppercase" }}>
              <i className="ti ti-database-search" style={{ fontSize: 12, marginRight: 5 }} aria-hidden="true" />Context
            </div>
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "5px 6px" }}>
            {sources.length === 0 && webResults.length === 0
              ? <div style={{ padding: "18px 8px", textAlign: "center", color: "var(--color-text-secondary)", fontSize: 11, lineHeight: 1.7 }}><i className="ti ti-search" style={{ fontSize: 20, display: "block", marginBottom: 5, opacity: 0.4 }} aria-hidden="true" />Retrieved chunks and web results appear here.</div>
              : <>
                {sources.length > 0 && (
                  <>
                    <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", padding: "4px 4px 4px 4px", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                      <i className="ti ti-file-text" style={{ fontSize: 11, marginRight: 4 }} aria-hidden="true" />Doc chunks
                    </div>
                    {sources.map((s, i) => (
                      <div key={i} style={{ padding: "7px 8px", marginBottom: 4, borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-secondary)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                          <span style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-warning)", background: "var(--color-background-warning)", padding: "1px 6px", borderRadius: 10 }}>#{i + 1}</span>
                          <span style={{ fontSize: 10, color: "var(--color-text-secondary)" }}>{s.score.toFixed(3)}</span>
                        </div>
                        <div style={{ fontSize: 11, fontWeight: 500, marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.docName}</div>
                        <div style={{ fontSize: 10, color: "var(--color-text-secondary)", lineHeight: 1.5, display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{s.text}</div>
                      </div>
                    ))}
                  </>
                )}
                {webResults.length > 0 && (
                  <>
                    <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", padding: "8px 4px 4px", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                      <i className="ti ti-search" style={{ fontSize: 11, marginRight: 4 }} aria-hidden="true" />Web results
                    </div>
                    {webResults.map((r, i) => (
                      <div key={i} style={{ padding: "7px 8px", marginBottom: 4, borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-success)", background: "var(--color-background-success)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                          <span style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-success)", background: "var(--color-background-success)", padding: "1px 6px", borderRadius: 10, border: "0.5px solid var(--color-border-success)" }}>W{i + 1}</span>
                        </div>
                        <div style={{ fontSize: 11, fontWeight: 500, marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={r.title}>{r.title || r.url}</div>
                        <div style={{ fontSize: 10, color: "var(--color-text-secondary)", marginBottom: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.url}</div>
                        <div style={{ fontSize: 10, color: "var(--color-text-secondary)", lineHeight: 1.5, display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{r.snippet}</div>
                      </div>
                    ))}
                  </>
                )}
              </>
            }
          </div>

          {/* Stats */}
          <div style={{ padding: "9px 10px", borderTop: "0.5px solid var(--color-border-tertiary)", flexShrink: 0 }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "var(--color-text-secondary)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 7 }}>
              <i className="ti ti-chart-bar" style={{ fontSize: 12, marginRight: 5 }} aria-hidden="true" />Stats
            </div>
            {[
              { label: "Docs", val: docs.length, icon: "ti-file-text" },
              { label: "Chunks", val: totalChunks, icon: "ti-puzzle" },
              { label: "Indexed", val: totalChars > 0 ? `${(totalChars / 1000).toFixed(1)}k` : "—", icon: "ti-ruler" },
              { label: "Turns", val: messages.filter(m => m.role === "user").length, icon: "ti-messages" },
            ].map(({ label, val, icon }) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                <span style={{ fontSize: 11, color: "var(--color-text-secondary)", display: "flex", alignItems: "center", gap: 5 }}>
                  <i className={`ti ${icon}`} style={{ fontSize: 12 }} aria-hidden="true" />{label}
                </span>
                <span style={{ fontSize: 11, fontWeight: 500 }}>{val}</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}