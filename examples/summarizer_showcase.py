"""
CCE Summarizer Showcase
Visual explanation of extractive summarization with D3 arc diagrams.
Run: uv run python examples/summarizer_showcase.py
Open: http://localhost:7862
"""

import re, sys, json, math
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

STOPWORDS = {
    "the","a","an","is","it","to","do","of","and","or","in","on","at","for",
    "with","that","this","was","are","be","have","has","had","you","your","we",
    "can","will","would","could","should","what","how","why","when","where",
    "about","just","so","but","if","not","no","i","me","my","also","its",
    "from","into","more","been","which","their","each","than","then",
    "them","these","they","used","using","use","while","across",
}
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

SAMPLE_TEXTS = {
    "fastapi": """FastAPI is a modern, high-performance web framework for building APIs with Python. It is based on standard Python type hints and provides automatic data validation. FastAPI generates interactive API documentation automatically using OpenAPI standards. The framework uses Pydantic for data validation and serialization of request and response models. Asynchronous programming with async and await allows FastAPI to handle many concurrent requests efficiently. PostgreSQL is the recommended database for production applications because it is reliable and feature-rich. SQLAlchemy provides an ORM layer that maps Python classes to database tables seamlessly. Alembic handles database migration scripts and tracks schema changes across environments. Authentication should use JWT tokens with OAuth2 for secure and stateless session management. Redis is commonly used for caching frequently accessed data and reducing database load significantly. Docker containers package the application and all its dependencies into a portable unit. Kubernetes orchestrates multiple containers and handles scaling, rolling updates, and self-healing. Monitoring with Sentry catches application errors while Datadog tracks performance metrics. Load balancers distribute incoming traffic across multiple server instances for high availability. CDNs serve static assets from edge locations closest to users for faster page load times.""",
    "machine_learning": """Machine learning is a subset of artificial intelligence that enables systems to learn from data. Supervised learning trains models on labeled datasets where each input has a known output value. Unsupervised learning discovers hidden patterns in unlabeled data without guidance. Reinforcement learning trains agents to make decisions by rewarding desired behaviors over time. Neural networks are computational models inspired by the structure of the human brain. Deep learning uses multiple layers of neural networks to learn hierarchical data representations. Gradient descent is the optimization algorithm used to minimize the loss function during training. Overfitting occurs when a model memorizes training data but fails to generalize to new inputs. Regularization techniques such as dropout and L2 penalty help prevent overfitting in neural networks. Cross-validation splits data into folds to evaluate model performance more reliably. Feature engineering transforms raw data into meaningful inputs that improve model accuracy significantly. Transfer learning reuses pretrained models to solve new tasks with less data and computation. Convolutional neural networks excel at image recognition by learning spatial feature hierarchies. Transformers revolutionized natural language processing using self-attention mechanisms across tokens. Embeddings map words and sentences into dense vectors that capture semantic meaning.""",
    "saas": """Building a SaaS product requires careful planning of the backend architecture from the start. A monolithic architecture is simpler to deploy and debug compared to microservices for early-stage products. PostgreSQL handles the majority of data storage needs for SaaS applications reliably and efficiently. Authentication should never be built from scratch because it introduces critical security vulnerabilities. Stripe is the industry standard payment processor and handles subscriptions and webhook events automatically. Multi-tenant database design uses a shared schema with a tenant identifier column on every table. React and Next.js are popular frontend choices because they support server-side rendering and routing. File uploads should be handled with presigned URLs that allow clients to upload directly to cloud storage. Background job queues prevent long-running tasks from blocking API response times and causing timeouts. Redis provides fast in-memory caching that reduces database load and improves response latency significantly. Docker containers ensure the application runs consistently across development and production environments. Monitoring requires tracking errors with Sentry, metrics with Datadog, and logs with structured JSON output. Database migrations must always be backward-compatible to support zero-downtime deployments safely. Rate limiting protects API endpoints from abuse and should be applied at the gateway level first. Horizontal scaling with load balancers distributes traffic across multiple instances for high availability."""
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
    # Build word-sentence connections for arc diagram
    connections = []
    for si, sent in enumerate(scored):
        for hit in sent["hits"]:
            connections.append({"word": hit, "sent_idx": si, "score": sent["score"]})
    return {
        "sentences": sents,
        "freq": freq_sorted[:12],
        "scored": scored,
        "selected": selected,
        "summary": " ".join(selected),
        "original_tokens": sum(max(1, round(len(s.split()) / 0.75)) for s in sents),
        "summary_tokens": used,
        "connections": connections,
    }


@app.get("/")
async def root():
    return HTMLResponse(open(Path(__file__).parent / "summarizer_showcase.html", encoding="utf-8").read())

@app.post("/analyze")
async def analyze(body: dict):
    text = body.get("text", "").strip()
    preset = body.get("preset", "")
    if preset and preset in SAMPLE_TEXTS:
        text = SAMPLE_TEXTS[preset]
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    result = run_summarizer(text)
    return {"result": result}

@app.get("/presets")
async def presets():
    return {k: v[:80] + "..." for k, v in SAMPLE_TEXTS.items()}

if __name__ == "__main__":
    print("\n  CCE Summarizer Showcase → http://localhost:7862\n")
    uvicorn.run(app, host="0.0.0.0", port=7862, log_level="warning")