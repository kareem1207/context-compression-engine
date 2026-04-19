"""
CCE Summarizer Showcase
Standalone page showing exactly how extractive summarization works —
step by step, with interactive pyvis graphs.

Run: uv run python examples/summarizer_showcase.py
Open: http://localhost:7862
"""

import re
import sys
import json
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyvis.network import Network
import tempfile, os, math

app = FastAPI()

STOPWORDS = {
    "the","a","an","is","it","to","do","of","and","or","in","on","at","for",
    "with","that","this","was","are","be","have","has","had","you","your","we",
    "can","will","would","could","should","what","how","why","when","where",
    "about","just","so","but","if","not","no","i","me","my","also","its",
    "are","from","into","more","been","which","their","each","than","then",
    "them","these","they","used","using","use","while","when","across",
}
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

SAMPLE_TEXTS = {
    "fastapi": """FastAPI is a modern, high-performance web framework for building APIs with Python.
It is based on standard Python type hints and provides automatic data validation.
FastAPI generates interactive API documentation automatically using OpenAPI standards.
The framework uses Pydantic for data validation and serialization of request and response models.
Asynchronous programming with async and await allows FastAPI to handle many concurrent requests efficiently.
PostgreSQL is the recommended database for production applications because it is reliable and feature-rich.
SQLAlchemy provides an ORM layer that maps Python classes to database tables seamlessly.
Alembic handles database migration scripts and tracks schema changes across environments.
Authentication should use JWT tokens with OAuth2 for secure and stateless session management.
Redis is commonly used for caching frequently accessed data and reducing database load significantly.
Docker containers package the application and all its dependencies into a portable unit.
Kubernetes orchestrates multiple containers and handles scaling, rolling updates, and self-healing.
Monitoring with Sentry catches application errors while Datadog tracks performance metrics.
Load balancers distribute incoming traffic across multiple server instances for high availability.
CDNs serve static assets from edge locations closest to users for faster page load times.""",

    "machine_learning": """Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning trains models on labeled datasets where each input has a known output value.
Unsupervised learning discovers hidden patterns and structures in unlabeled data without guidance.
Reinforcement learning trains agents to make decisions by rewarding desired behaviors over time.
Neural networks are computational models inspired by the structure of the human brain.
Deep learning uses multiple layers of neural networks to learn hierarchical data representations.
Gradient descent is the optimization algorithm used to minimize the loss function during training.
Overfitting occurs when a model memorizes training data but fails to generalize to new inputs.
Regularization techniques such as dropout and L2 penalty help prevent overfitting in neural networks.
Cross-validation splits data into folds to evaluate model performance more reliably.
Feature engineering transforms raw data into meaningful inputs that improve model accuracy significantly.
Transfer learning reuses pretrained models to solve new tasks with less data and computation.
Convolutional neural networks excel at image recognition by learning spatial feature hierarchies.
Transformers revolutionized natural language processing using self-attention mechanisms across tokens.
Embeddings map words and sentences into dense numerical vectors that capture semantic meaning.""",

    "saas": """Building a SaaS product requires careful planning of the backend architecture from the start.
A monolithic architecture is simpler to deploy and debug compared to microservices for early-stage products.
PostgreSQL handles the majority of data storage needs for SaaS applications reliably and efficiently.
Authentication should never be built from scratch because it introduces critical security vulnerabilities.
Stripe is the industry standard payment processor and handles subscriptions and webhook events automatically.
Multi-tenant database design uses a shared schema with a tenant identifier column on every table.
React and Next.js are popular frontend choices because they support server-side rendering and routing.
File uploads should be handled with presigned URLs that allow clients to upload directly to cloud storage.
Background job queues prevent long-running tasks from blocking API response times and causing timeouts.
Redis provides fast in-memory caching that reduces database load and improves response latency significantly.
Docker containers ensure the application runs consistently across development, staging, and production environments.
Monitoring requires tracking errors with Sentry, metrics with Datadog, and logs with structured JSON output.
Database migrations must always be backward-compatible to support zero-downtime deployments safely.
Rate limiting protects API endpoints from abuse and should be applied at the gateway level first.
Horizontal scaling with load balancers distributes traffic across multiple instances for high availability."""
}


def run_summarizer(text: str, max_tokens: int = 200) -> dict:
    sents = [s.strip() for s in SENT_RE.split(text.strip()) if s.strip()]
    if not sents:
        sents = [text]

    words = WORD_RE.findall(text.lower())
    freq = Counter(w for w in words if w not in STOPWORDS)
    top_words = {w: c for w, c in freq.most_common(14)}
    freq_sorted = sorted(top_words.items(), key=lambda x: -x[1])

    scored = []
    for s in sents:
        ws = set(WORD_RE.findall(s.lower())) - STOPWORDS
        if not ws:
            scored.append({"text": s, "score": 0.0, "hits": []})
            continue
        overlap = len(ws & set(top_words)) / len(ws)
        length_bonus = min(len(s.split()) / 20.0, 1.0) * 0.2
        score = round(overlap + length_bonus, 3)
        scored.append({"text": s, "score": score, "hits": sorted(ws & set(top_words))})

    scored_sorted = sorted(scored, key=lambda x: -x["score"])
    selected, used = [], 0
    for item in scored_sorted:
        tok = max(1, round(len(item["text"].split()) / 0.75))
        if used + tok > max_tokens:
            break
        selected.append(item["text"])
        used += tok

    order = {s: i for i, s in enumerate(sents)}
    selected.sort(key=lambda s: order.get(s, 999))

    return {
        "sentences": sents,
        "freq": freq_sorted[:12],
        "scored": scored,
        "selected": selected,
        "summary": " ".join(selected),
        "original_tokens": sum(max(1, round(len(s.split()) / 0.75)) for s in sents),
        "summary_tokens": used,
    }


def make_cooc_graph(freq_pairs: list, sentences: list) -> str:
    net = Network(height="380px", width="100%", bgcolor="#0f1117", font_color="#e2e8f0")
    net.set_options(json.dumps({
        "physics": {"stabilization": {"iterations": 120}, "barnesHut": {"gravitationalConstant": -3000}},
        "nodes": {"borderWidth": 0, "shadow": False},
        "edges": {"shadow": False, "smooth": {"type": "continuous"}},
        "interaction": {"tooltipDelay": 100}
    }))
    words = [p[0] for p in freq_pairs]
    counts = {p[0]: p[1] for p in freq_pairs}
    maxc = max(counts.values()) if counts else 1
    for w in words:
        size = 12 + int((counts[w] / maxc) * 28)
        r = int(80 - (counts[w] / maxc) * 20)
        g = int(180 + (counts[w] / maxc) * 31)
        color = f"#{r:02x}{g:02x}99"
        net.add_node(w, label=w, size=size, color=color,
                     title=f"<b>{w}</b><br>appears {counts[w]} times")
    added = set()
    for sent in sentences:
        ws = [w for w in words if w in sent.lower()]
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                key = tuple(sorted([ws[i], ws[j]]))
                if key not in added:
                    net.add_edge(ws[i], ws[j], color="#2d3748", width=1.5)
                    added.add(key)
    return _extract_body(net)


def make_sim_graph(scored: list) -> str:
    net = Network(height="380px", width="100%", bgcolor="#0f1117", font_color="#e2e8f0")
    net.set_options(json.dumps({
        "physics": {"stabilization": {"iterations": 120}, "barnesHut": {"gravitationalConstant": -2500}},
        "nodes": {"borderWidth": 0, "shape": "dot"},
        "edges": {"shadow": False, "smooth": {"type": "continuous"}},
        "interaction": {"tooltipDelay": 100}
    }))
    items = scored[:12]
    maxs = max(s["score"] for s in items) if items else 1
    for i, s in enumerate(items):
        pct = s["score"] / maxs if maxs > 0 else 0
        r = int(255 * (1 - pct))
        g = int(200 * pct)
        color = f"#{min(r,255):02x}{min(g,255):02x}80"
        size = 10 + int(pct * 24)
        label = s["text"][:30] + ("..." if len(s["text"]) > 30 else "")
        title = f"<b>Score: {s['score']}</b><br>{s['text']}<br>Keywords: {', '.join(s['hits'][:5])}"
        net.add_node(i, label=label, size=size, color=color, title=title)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            si = set(WORD_RE.findall(items[i]["text"].lower())) - STOPWORDS
            sj = set(WORD_RE.findall(items[j]["text"].lower())) - STOPWORDS
            if si and sj:
                sim = len(si & sj) / math.sqrt(len(si) * len(sj))
                if sim > 0.12:
                    net.add_edge(i, j, value=round(sim, 2), color="#2d3748",
                                 title=f"similarity: {sim:.2f}", width=sim * 3)
    return _extract_body(net)


def _extract_body(net: Network) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    tmp_path = tmp.name
    tmp.close()  # must close before pyvis writes on Windows
    net.save_graph(tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        html = f.read()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass  # Windows holds handle briefly — not critical
    start = html.find("<body>") + 6
    end = html.find("</body>")
    return html[start:end].strip()


@app.get("/")
async def root():
    return HTMLResponse(SHOWCASE_HTML)


@app.post("/analyze")
async def analyze(body: dict):
    text = body.get("text", "").strip()
    preset = body.get("preset", "")
    if preset and preset in SAMPLE_TEXTS:
        text = SAMPLE_TEXTS[preset]
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    result = run_summarizer(text)
    cooc = make_cooc_graph(result["freq"], result["sentences"])
    sim = make_sim_graph(result["scored"])
    return {"result": result, "graphs": {"cooc": cooc, "sim": sim}}


@app.get("/presets")
async def presets():
    return {k: v[:80] + "..." for k, v in SAMPLE_TEXTS.items()}


SHOWCASE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CCE Summarizer Showcase</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;500;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0e14;--bg2:#111520;--bg3:#181e2a;--bg4:#1e2535;
  --border:#1e2736;--border2:#263040;
  --text:#e2e8f8;--muted:#5a7090;--dim:#2a3548;
  --blue:#4f9eff;--purple:#8b7fff;--green:#34d399;--amber:#fbbf24;--red:#f87171;--teal:#2dd4bf;
  --font:'Syne',sans-serif;--mono:'JetBrains Mono',monospace;
}
html,body{min-height:100%;background:var(--bg);color:var(--text);font-family:var(--font)}
body{display:flex;flex-direction:column;overflow-x:hidden}

header{padding:16px 28px;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:14px}
.hbadge{font-size:9px;font-family:var(--mono);letter-spacing:2px;color:var(--muted);border:1px solid var(--border2);padding:3px 10px;border-radius:20px}
.htitle{font-size:18px;font-weight:700;letter-spacing:-.3px}
.hsub{font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:2px}

.main{flex:1;display:grid;grid-template-columns:340px 1fr;gap:0;overflow:hidden;min-height:calc(100vh - 60px)}

/* LEFT: input + controls */
.left{background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow-y:auto}
.sect{padding:16px 18px;border-bottom:1px solid var(--border)}
.sect-title{font-size:10px;font-family:var(--mono);color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-bottom:10px}
.preset-grid{display:flex;flex-direction:column;gap:6px}
.preset-btn{padding:9px 12px;border-radius:8px;border:1px solid var(--border2);background:var(--bg3);color:var(--muted);font-size:12px;font-family:var(--font);cursor:pointer;text-align:left;transition:all .15s;line-height:1.4}
.preset-btn:hover,.preset-btn.active{border-color:var(--blue);color:var(--text);background:rgba(79,158,255,.08)}
.preset-name{font-weight:600;font-size:12px;margin-bottom:2px}
.preset-preview{font-size:10px;color:var(--muted);font-family:var(--mono)}
textarea.tinp{width:100%;background:var(--bg3);border:1px solid var(--border2);border-radius:8px;padding:10px 12px;font-size:12px;color:var(--text);font-family:var(--mono);resize:vertical;outline:none;line-height:1.6;min-height:120px}
textarea.tinp:focus{border-color:var(--blue)}
.run-btn{width:100%;padding:12px;border-radius:8px;background:var(--blue);border:none;color:#000;font-size:13px;font-weight:700;font-family:var(--font);cursor:pointer;transition:all .15s;letter-spacing:.3px;margin-top:10px}
.run-btn:hover{background:#6fb3ff;transform:translateY(-1px)}
.run-btn:disabled{opacity:.4;cursor:default;transform:none}
.token-info{background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:11px;font-family:var(--mono);color:var(--muted);line-height:1.9;margin-top:10px;display:none}
.token-info b{color:var(--green)}

/* RIGHT: steps + graphs */
.right{overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:18px}
.right::-webkit-scrollbar{width:4px}
.right::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

.step-card{background:var(--bg2);border:1px solid var(--border2);border-radius:12px;overflow:hidden;animation:slideIn .35s ease}
@keyframes slideIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.step-hdr{display:flex;align-items:center;gap:10px;padding:12px 16px;background:var(--bg3);border-bottom:1px solid var(--border)}
.step-num{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;font-family:var(--mono);flex-shrink:0}
.n1{background:rgba(79,158,255,.2);color:var(--blue);border:1px solid var(--blue)}
.n2{background:rgba(52,211,153,.2);color:var(--green);border:1px solid var(--green)}
.n3{background:rgba(251,191,36,.2);color:var(--amber);border:1px solid var(--amber)}
.n4{background:rgba(139,127,255,.2);color:var(--purple);border:1px solid var(--purple)}
.step-title{font-size:14px;font-weight:600}
.step-sub{font-size:10px;color:var(--muted);font-family:var(--mono);margin-top:1px}
.step-body{padding:14px 16px}

/* Step 1: sentences */
.sent-list{display:flex;flex-direction:column;gap:5px}
.sent-item{padding:8px 12px;border-radius:6px;font-size:12px;line-height:1.6;border-left:3px solid var(--border2);background:var(--bg3);color:var(--muted);transition:all .2s;position:relative}
.sent-item.selected{border-left-color:var(--green);background:rgba(52,211,153,.06);color:var(--text)}
.sent-badge{position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:9px;font-family:var(--mono);padding:2px 6px;border-radius:3px;background:rgba(52,211,153,.2);color:var(--green)}

/* Step 2: freq */
.freq-wrap{display:flex;flex-wrap:wrap;gap:7px}
.freq-chip{padding:5px 12px;border-radius:20px;font-size:11px;font-family:var(--mono);border:1px solid transparent;display:flex;align-items:center;gap:6px;cursor:default}
.freq-count{font-size:9px;opacity:.7}

/* Step 3: scored */
.scored-list{display:flex;flex-direction:column;gap:6px}
.scored-row{display:flex;gap:12px;align-items:flex-start;padding:8px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg3)}
.score-bar-col{display:flex;flex-direction:column;align-items:center;gap:4px;flex-shrink:0;width:36px}
.score-bar-track{width:8px;height:60px;background:var(--border);border-radius:4px;overflow:hidden;display:flex;flex-direction:column;justify-content:flex-end}
.score-bar-fill{width:100%;border-radius:4px;transition:height .5s ease}
.score-val{font-size:9px;font-family:var(--mono);color:var(--muted)}
.scored-right{flex:1;min-width:0}
.scored-text{font-size:11px;color:var(--muted);line-height:1.5;margin-bottom:5px}
.kw-chips{display:flex;flex-wrap:wrap;gap:4px}
.kw-chip{font-size:9px;padding:2px 6px;border-radius:3px;background:rgba(52,211,153,.1);color:var(--green);font-family:var(--mono)}

/* Step 4: selected + summary */
.selected-list{display:flex;flex-direction:column;gap:5px;margin-bottom:12px}
.final-box{background:rgba(79,158,255,.07);border:1px solid rgba(79,158,255,.3);border-radius:8px;padding:12px 14px;font-size:13px;color:var(--text);line-height:1.7}
.final-label{font-size:10px;font-family:var(--mono);color:var(--muted);margin-bottom:6px}

/* Graphs */
.graphs-row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.graph-card{background:var(--bg2);border:1px solid var(--border2);border-radius:12px;overflow:hidden}
.graph-title{padding:10px 14px;font-size:12px;font-weight:600;border-bottom:1px solid var(--border);background:var(--bg3)}
.graph-sub{font-size:10px;color:var(--muted);font-family:var(--mono);margin-top:2px}
.graph-body{background:#0f1117}
.loading-state{text-align:center;padding:40px;font-size:11px;color:var(--muted);font-family:var(--mono)}

.empty-state{text-align:center;padding:60px 20px;color:var(--muted);font-size:13px;line-height:1.8}
.empty-state .big{font-size:32px;margin-bottom:12px;opacity:.3}
</style>
</head>
<body>

<header>
  <div class="hbadge">CCE</div>
  <div>
    <div class="htitle">Summarizer Showcase</div>
    <div class="hsub">Watch extractive summarization work step by step · with interactive graphs</div>
  </div>
</header>

<div class="main">
  <!-- LEFT -->
  <div class="left">
    <div class="sect">
      <div class="sect-title">Sample texts</div>
      <div class="preset-grid" id="presetGrid">
        <div class="preset-btn active" onclick="loadPreset('fastapi')">
          <div class="preset-name">FastAPI + Backend</div>
          <div class="preset-preview">FastAPI, PostgreSQL, Docker, Redis...</div>
        </div>
        <div class="preset-btn" onclick="loadPreset('machine_learning')">
          <div class="preset-name">Machine Learning</div>
          <div class="preset-preview">Supervised, neural networks, transformers...</div>
        </div>
        <div class="preset-btn" onclick="loadPreset('saas')">
          <div class="preset-name">SaaS Architecture</div>
          <div class="preset-preview">Auth, Stripe, Redis, Docker, scaling...</div>
        </div>
      </div>
    </div>

    <div class="sect" style="flex:1">
      <div class="sect-title">Or paste your own text</div>
      <textarea class="tinp" id="textInput" placeholder="Paste any multi-sentence paragraph here..."></textarea>
      <button class="run-btn" id="runBtn" onclick="runAnalysis()">▶ Run Summarizer</button>
      <div class="token-info" id="tokenInfo"></div>
    </div>
  </div>

  <!-- RIGHT -->
  <div class="right" id="rightPanel">
    <div class="empty-state">
      <div class="big">◈</div>
      Select a sample text on the left and click <b>▶ Run Summarizer</b><br>
      to watch all 4 steps of extractive summarization<br>with interactive word graphs
    </div>
  </div>
</div>

<script>
var PRESETS = {
  fastapi: null,
  machine_learning: null,
  saas: null
};
var activePreset = 'fastapi';
var running = false;

// Load preset texts from server
fetch('/presets').then(function(r){return r.json()}).then(function(d){
  PRESETS = d;
  document.getElementById('textInput').value = '';
});

function loadPreset(name) {
  activePreset = name;
  document.querySelectorAll('.preset-btn').forEach(function(b){b.classList.remove('active')});
  event.currentTarget.classList.add('active');
  document.getElementById('textInput').value = '';
  document.getElementById('textInput').placeholder = 'Using preset: ' + name + '\n(or paste your own text to override)';
}

async function runAnalysis() {
  if (running) return;
  running = true;
  var btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Analyzing...';

  var userText = document.getElementById('textInput').value.trim();
  var payload = userText ? {text: userText} : {preset: activePreset};

  var panel = document.getElementById('rightPanel');
  panel.innerHTML = '<div class="loading-state">Running summarizer...</div>';

  try {
    var resp = await fetch('/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    var data = await resp.json();
    if (data.error) throw new Error(data.error);
    await renderResults(data.result, data.graphs);
  } catch(e) {
    panel.innerHTML = '<div class="loading-state" style="color:var(--red)">Error: ' + e.message + '</div>';
  }

  btn.disabled = false;
  btn.textContent = '↺ Run Again';
  running = false;
}

async function renderResults(r, graphs) {
  var panel = document.getElementById('rightPanel');
  panel.innerHTML = '';

  // Token info
  var info = document.getElementById('tokenInfo');
  info.style.display = 'block';
  info.innerHTML =
    'Sentences: <b>' + r.sentences.length + '</b> · ' +
    'Original: <b>' + r.original_tokens + ' tokens</b> · ' +
    'Summary: <b>' + r.summary_tokens + ' tokens</b> · ' +
    'Compression: <b>' + Math.round(r.original_tokens / Math.max(r.summary_tokens,1)) + 'x</b>';

  var selSet = new Set(r.selected);
  var maxScore = r.scored.length ? Math.max.apply(null, r.scored.map(function(x){return x.score})) : 1;
  var maxFreq = r.freq.length ? r.freq[0][1] : 1;

  // ── Step 1 ─────────────────────────────────────────────────────────────────
  await delay(0);
  var s1 = makeStepCard(1, 'n1', 'Split into sentences', r.sentences.length + ' sentences found using punctuation boundaries');
  var list1 = div('sent-list');
  r.sentences.forEach(function(s) {
    var el = div('sent-item' + (selSet.has(s) ? ' selected' : ''));
    el.textContent = s;
    if (selSet.has(s)) {
      var badge = document.createElement('span');
      badge.className = 'sent-badge';
      badge.textContent = '✓ selected';
      el.appendChild(badge);
    }
    list1.appendChild(el);
  });
  s1.body.appendChild(list1);
  panel.appendChild(s1.card);
  panel.scrollTop = panel.scrollHeight;

  // ── Step 2 ─────────────────────────────────────────────────────────────────
  await delay(500);
  var s2 = makeStepCard(2, 'n2', 'Build word frequency map', 'Stopwords removed · keyword importance by count · bigger chip = more frequent');
  var fw = div('freq-wrap');
  r.freq.forEach(function(p) {
    var pct = p[1] / maxFreq;
    var chip = div('freq-chip');
    chip.style.background = 'rgba(52,211,153,' + (0.08 + pct * 0.45) + ')';
    chip.style.borderColor = 'rgba(52,211,153,' + (0.15 + pct * 0.4) + ')';
    chip.style.color = 'var(--green)';
    chip.style.fontSize = (10 + Math.round(pct * 4)) + 'px';
    chip.innerHTML = esc(p[0]) + ' <span class="freq-count">×' + p[1] + '</span>';
    fw.appendChild(chip);
  });
  s2.body.appendChild(fw);
  panel.appendChild(s2.card);
  panel.scrollTop = panel.scrollHeight;

  // ── Step 3 ─────────────────────────────────────────────────────────────────
  await delay(500);
  var s3 = makeStepCard(3, 'n3', 'Score each sentence', 'Score = keyword overlap ÷ sentence words + length bonus · green bar = higher score');
  var sl = div('scored-list');
  r.scored.forEach(function(item) {
    var pct = maxScore > 0 ? item.score / maxScore : 0;
    var col = pct > 0.6 ? 'var(--green)' : pct > 0.35 ? 'var(--amber)' : 'var(--red)';
    var row = div('scored-row');
    var barCol = div('score-bar-col');
    barCol.innerHTML =
      '<div class="score-bar-track"><div class="score-bar-fill" style="height:' + Math.round(pct*100) + '%;background:' + col + '"></div></div>' +
      '<div class="score-val">' + item.score + '</div>';
    var right = div('scored-right');
    right.innerHTML =
      '<div class="scored-text">' + esc(item.text.substring(0,90)) + (item.text.length>90?'...':'') + '</div>' +
      (item.hits.length ? '<div class="kw-chips">' + item.hits.slice(0,6).map(function(h){return '<span class="kw-chip">'+esc(h)+'</span>'}).join('') + '</div>' : '');
    row.appendChild(barCol);
    row.appendChild(right);
    sl.appendChild(row);
  });
  s3.body.appendChild(sl);
  panel.appendChild(s3.card);
  panel.scrollTop = panel.scrollHeight;

  // ── Step 4 ─────────────────────────────────────────────────────────────────
  await delay(500);
  var s4 = makeStepCard(4, 'n4', 'Pick top sentences', 'Greedy selection within token budget · re-ordered by original position');
  var sl2 = div('selected-list');
  r.selected.forEach(function(s) {
    var el = div('sent-item selected');
    el.textContent = s;
    sl2.appendChild(el);
  });
  var fl = document.createElement('div');
  fl.className = 'final-label';
  fl.textContent = 'Meso summary (' + r.summary_tokens + ' tokens):';
  var fb = div('final-box');
  fb.textContent = r.summary;
  s4.body.appendChild(sl2);
  s4.body.appendChild(fl);
  s4.body.appendChild(fb);
  panel.appendChild(s4.card);
  panel.scrollTop = panel.scrollHeight;

  // ── Graphs ─────────────────────────────────────────────────────────────────
  await delay(300);
  var gRow = div('graphs-row');

  var gc = div('graph-card');
  gc.innerHTML = '<div class="graph-title">Word co-occurrence graph<div class="graph-sub">Nodes = keywords · size = frequency · edges = appear in same sentence · drag to explore</div></div><div class="graph-body">' + graphs.cooc + '</div>';

  var gs = div('graph-card');
  gs.innerHTML = '<div class="graph-title">Sentence similarity graph<div class="graph-sub">Nodes = sentences · green = high score · red = low score · edges = shared keywords</div></div><div class="graph-body">' + graphs.sim + '</div>';

  gRow.appendChild(gc);
  gRow.appendChild(gs);
  panel.appendChild(gRow);
  panel.scrollTop = panel.scrollHeight;
}

// Helpers
function delay(ms) { return new Promise(function(r){setTimeout(r,ms)}); }
function div(cls) { var d=document.createElement('div'); d.className=cls; return d; }
function makeStepCard(n, numCls, title, sub) {
  var card = div('step-card');
  var hdr = div('step-hdr');
  var num = div('step-num ' + numCls);
  num.textContent = n;
  var info = div('');
  info.innerHTML = '<div class="step-title">' + esc(title) + '</div><div class="step-sub">' + esc(sub) + '</div>';
  hdr.appendChild(num); hdr.appendChild(info);
  var body = div('step-body');
  card.appendChild(hdr); card.appendChild(body);
  return {card: card, body: body};
}
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("\n  CCE Summarizer Showcase → http://localhost:7862\n")
    uvicorn.run(app, host="0.0.0.0", port=7862, log_level="warning")