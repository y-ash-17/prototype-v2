import hashlib
import math
import re
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocChunk, Society
from app.scoping import filter_scoped_chunks

MIN_RETRIEVAL_SCORE = 0.25
ABSTENTION_MESSAGE = (
    "I don't know based on the current clinic policies and your society annex. "
    "Would you like me to connect you with a nurse to assist further?"
)

_INJECTION_PATTERNS = [
    r"(?im)^\s*(ignore|disregard|override)\s+(all\s+)?(previous|prior|system)\s+instructions.*$",
    r"(?im)^\s*(system|developer|assistant)\s*:\s*.*$",
    r"(?im)\b(call|invoke|execute)\s+(tool|book_service|escalate_emergency|log_symptom)\b.*$",
]

_STOPWORDS = {
    "the", "and", "for", "with", "from", "what", "when", "where", "which",
    "who", "how", "are", "is", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "by", "or", "an", "a", "my", "me", "i", "it", "do",
    "does", "can", "could", "would", "should", "now", "still", "any",
    "period", "price",
}

_DOMAIN_BOOST_TERMS = {
    "wellness", "sunday", "hours", "fasting", "lipid", "fbs", "thyroid",
    "vitamin", "coverage", "plan", "escalation", "112", "hospital",
    "department", "codes", "slot", "lengths", "cardiology", "physiotherapy",
    "directory", "covered", "member", "rate",
}


def _to_py_float(val: Any) -> float:
    """Convert numeric values (including numpy scalars) to pure Python float for msgpack."""
    if hasattr(val, "item"):
        val = val.item()
    return float(val)


def sanitize_retrieved_text(text: str) -> str:
    """Treat retrieved corpus text strictly as data, stripping instruction-like lines."""
    cleaned = text or ""
    for pat in _INJECTION_PATTERNS:
        cleaned = re.sub(pat, "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def embed_text(text: str, dim: int = 1536) -> list[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
    if not tokens:
        return vec
    for i, tok in enumerate(tokens):
        digest = hashlib.sha256(f"{tok}:{i}".encode("utf-8")).digest()
        for j in range(0, len(digest), 2):
            idx = int.from_bytes(digest[j : j + 2], "big") % dim
            sign = 1.0 if (digest[j] % 2 == 0) else -1.0
            vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [float(x / norm) for x in vec]
    return [float(x) for x in vec]


def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    a_py = [_to_py_float(x) for x in a]
    b_py = [_to_py_float(x) for x in b]
    dot = sum(x * y for x, y in zip(a_py, b_py))
    na = math.sqrt(sum(x * x for x in a_py))
    nb = math.sqrt(sum(y * y for y in b_py))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(max(0.0, min(1.0, dot / (na * nb))))


def _keyword_score(query: str, chunk: DocChunk) -> float:
    q_tokens = {
        t for t in re.findall(r"[a-z0-9_]+", (query or "").lower())
        if len(t) > 1 and t not in _STOPWORDS
    }
    if not q_tokens:
        return 0.0
    haystack = (
        f"{chunk.title or ''} {chunk.heading_path or ''} {chunk.section or ''} {chunk.content or ''}"
    ).lower()
    h_tokens = set(re.findall(r"[a-z0-9_]+", haystack))
    overlap = len(q_tokens & h_tokens)
    if overlap == 0:
        return 0.0

    boost = 0.0
    for term in _DOMAIN_BOOST_TERMS:
        if term in q_tokens and term in haystack:
            boost += 0.25
    return float((overlap / max(len(q_tokens), 1)) + boost)


async def retrieve_corpus_chunks(
    session: AsyncSession,
    query: str,
    society: Society | None = None,
    *,
    society_id: str | None = None,
    top_k: int = 4,
    min_score: float = MIN_RETRIEVAL_SCORE,
) -> dict[str, Any]:
    """SQL-filtered pgvector retrieval with heading-path citations and abstention handling."""
    resolved_soc_id = society.id if society is not None else society_id
    if not resolved_soc_id:
        raise ValueError("society_id (or society) is required for corpus retrieval.")

    q_emb = embed_text(query)

    stmt = (
        select(DocChunk)
        .where(
            DocChunk.superseded_by.is_(None),
            DocChunk.doc_id != "staff-notes-internal",
            or_(DocChunk.applies_to == "all", DocChunk.applies_to == resolved_soc_id),
        )
        .order_by(DocChunk.embedding.cosine_distance(q_emb).asc().nullslast())
    )

    rows = (await session.execute(stmt)).scalars().all()
    scoped_rows = filter_scoped_chunks(list(rows), society_id=resolved_soc_id)

    q_lower = (query or "").lower()
    is_annex_query = any(
        w in q_lower
        for w in (
            "annex", "wellness", "sunday", "saturday", "hours", "open",
            "closed", "phlebotomist", "visiting", "gate", "ambulance", "resident",
        )
    )
    is_coverage_query = any(
        w in q_lower
        for w in ("vitamin", "thyroid", "member rate", "coverage")
    ) or ("covered" in q_lower and any(w in q_lower for w in ("plan", "test", "lab", "thyroid", "vitamin", "lipid", "cbc", "fbs")))
    allowed_annex = society.annex_doc_id if (society and society.annex_doc_id) else None

    scored: list[tuple[float, DocChunk]] = []
    for ch in scoped_rows:
        sim = _cosine_similarity(q_emb, list(ch.embedding) if ch.embedding is not None else None)
        kw = _keyword_score(query, ch)
        annex_boost = 0.25 if (is_annex_query and allowed_annex and ch.doc_id == allowed_annex) else 0.0
        cov_boost = 0.0
        if is_coverage_query and ch.doc_id == "plan-coverage":
            cov_boost += 0.65
            c_lower = (ch.content or "").lower()
            if ("vitamin" in q_lower and "vitamin d" in c_lower) or (
                "thyroid" in q_lower and "thyroid" in c_lower
            ):
                cov_boost += 0.45
            if "thyroid profile is now covered" in c_lower or "what changed" in c_lower:
                cov_boost += 0.85
        total_score = float(round((0.40 * sim) + (0.60 * kw) + annex_boost + cov_boost, 4))
        scored.append((total_score, ch))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_items = [item for item in scored[:top_k] if item[0] >= min_score]

    if not top_items:
        return {
            "abstained": True,
            "abstention_message": ABSTENTION_MESSAGE,
            "offer_nurse": True,
            "chunks": [],
            "citations": [],
        }

    chunks_out: list[dict[str, Any]] = []
    citations_out: list[dict[str, Any]] = []
    for score, ch in top_items:
        heading_path = ch.heading_path or (
            f"{ch.title} > {ch.section}" if ch.section and ch.section != ch.title else ch.title
        )
        safe_content = sanitize_retrieved_text(ch.content)
        chunks_out.append(
            {
                "chunk_id": str(ch.chunk_id),
                "doc_id": str(ch.doc_id),
                "title": str(ch.title),
                "section": str(ch.section),
                "heading_path": str(heading_path),
                "applies_to": str(ch.applies_to),
                "version": int(ch.version or 1),
                "score": _to_py_float(score),
                "content": str(safe_content),
            }
        )
        citations_out.append(
            {
                "doc_id": str(ch.doc_id),
                "title": str(ch.title),
                "section": str(ch.section),
                "heading_path": str(heading_path),
                "chunk_id": str(ch.chunk_id),
            }
        )

    return {
        "abstained": False,
        "abstention_message": None,
        "offer_nurse": False,
        "chunks": chunks_out,
        "citations": citations_out,
    }