import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from sqlalchemy import delete, select

from app.db import AsyncSessionLocal
from app.models import Clinician, DocChunk, Document, Patient, Society

ROOT_DIR = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT_DIR / "data" / "seed"
CORPUS_DIR = ROOT_DIR / "data" / "corpus"


def parse_frontmatter_and_sections(md_text: str, doc_id: str) -> tuple[dict, list[dict]]:
    meta: dict = {
        "doc_id": doc_id,
        "title": doc_id,
        "applies_to": "all",
        "version": 1,
        "effective_from": None,
        "superseded_by": None,
    }
    body = md_text

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", md_text, flags=re.DOTALL)
    if fm_match:
        fm_block, body = fm_match.group(1), fm_match.group(2)
        for line in fm_block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if val.lower() == "null" or val == "":
                    val = None
                elif key == "version":
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                meta[key] = val

    parts = re.split(r"(?m)^(#{1,3}\s+.+)$", body.strip())
    chunks: list[dict] = []
    heading_stack: list[tuple[int, str]] = []
    current_section = meta["title"]
    current_heading_path = meta["title"]
    buffer: list[str] = []

    def flush_chunk(idx: int, section: str, heading_path: str, content: str) -> dict | None:
        cleaned = content.strip()
        if not cleaned:
            return None
        return {
            "chunk_id": f"{meta['doc_id']}#chunk-{idx}",
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "section": section,
            "heading_path": heading_path,
            "applies_to": meta["applies_to"],
            "version": meta["version"],
            "effective_from": meta["effective_from"],
            "superseded_by": meta["superseded_by"],
            "content": cleaned,
        }

    idx = 0
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        h_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if h_match:
            flushed = flush_chunk(idx, current_section, current_heading_path, "\n".join(buffer))
            if flushed:
                chunks.append(flushed)
                idx += 1
            level = len(h_match.group(1))
            heading_text = h_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_section = heading_text
            if heading_stack[0][1].lower() == meta["title"].lower():
                current_heading_path = " > ".join(h for _, h in heading_stack)
            else:
                current_heading_path = f"{meta['title']} > " + " > ".join(h for _, h in heading_stack)
            buffer = [stripped]
        else:
            buffer.append(stripped)

    flushed = flush_chunk(idx, current_section, current_heading_path, "\n".join(buffer))
    if flushed:
        chunks.append(flushed)

    deduped: dict[str, dict] = {}
    for ch in chunks:
        cid = ch["chunk_id"]
        if cid in deduped:
            deduped[cid]["content"] = f"{deduped[cid]['content']}\n\n{ch['content']}".strip()
        else:
            deduped[cid] = ch

    return meta, list(deduped.values())


def deterministic_embedding(text: str, dim: int = 1536) -> list[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return vec
    for i, tok in enumerate(tokens):
        digest = hashlib.sha256(f"{tok}:{i}".encode("utf-8")).digest()
        for j in range(0, len(digest), 2):
            idx = int.from_bytes(digest[j : j + 2], "big") % dim
            sign = 1.0 if (digest[j] % 2 == 0) else -1.0
            vec[idx] += sign

    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


async def seed_all() -> None:
    async with AsyncSessionLocal() as session:
        societies_path = SEED_DIR / "societies.json"
        if societies_path.exists():
            for item in json.loads(societies_path.read_text(encoding="utf-8")):
                existing = await session.get(Society, item["id"])
                if existing:
                    for k, v in item.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(Society(**item))

        clinicians_path = SEED_DIR / "clinicians.json"
        if clinicians_path.exists():
            for item in json.loads(clinicians_path.read_text(encoding="utf-8")):
                existing = await session.get(Clinician, item["id"])
                if existing:
                    for k, v in item.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(Clinician(**item))

        patients_path = SEED_DIR / "patients.json"
        if patients_path.exists():
            for item in json.loads(patients_path.read_text(encoding="utf-8")):
                existing = await session.get(Patient, item["id"])
                if existing:
                    for k, v in item.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(Patient(**item))

        await session.flush()

        if CORPUS_DIR.exists():
            seen_doc_ids: set[str] = set()
            seen_chunk_ids: set[str] = set()
            for md_file in sorted(CORPUS_DIR.glob("*.md")):
                file_stem = md_file.stem
                raw_text = md_file.read_text(encoding="utf-8")
                meta, chunks = parse_frontmatter_and_sections(raw_text, file_stem)
                doc_id = meta.get("doc_id") or file_stem
                # Only ingest active versions; otherwise older chunks win duplicate IDs.
                if meta.get("superseded_by"):
                    continue
                await session.execute(delete(DocChunk).where(
                    DocChunk.doc_id == doc_id, DocChunk.version != meta["version"]
                ))

                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    existing_doc = await session.get(Document, doc_id)
                    if existing_doc:
                        existing_doc.title = meta["title"]
                        existing_doc.applies_to = meta["applies_to"]
                        existing_doc.version = meta["version"]
                        existing_doc.effective_from = meta["effective_from"]
                        existing_doc.superseded_by = meta["superseded_by"]
                        existing_doc.content = raw_text
                    else:
                        session.add(
                            Document(
                                doc_id=doc_id,
                                title=meta["title"],
                                applies_to=meta["applies_to"],
                                version=meta["version"],
                                effective_from=meta["effective_from"],
                                superseded_by=meta["superseded_by"],
                                content=raw_text,
                            )
                        )
                    await session.flush()
                else:
                    # If a later file (e.g., plan-coverage-v2.md) has a higher version for the same doc_id,
                    # update the already-stashed or existing Document record.
                    existing_doc = await session.get(Document, doc_id)
                    if existing_doc and (meta.get("version") or 1) >= (existing_doc.version or 1):
                        existing_doc.title = meta["title"]
                        existing_doc.applies_to = meta["applies_to"]
                        existing_doc.version = meta["version"]
                        existing_doc.effective_from = meta["effective_from"]
                        existing_doc.superseded_by = meta["superseded_by"]
                        existing_doc.content = raw_text
                        await session.flush()

                for ch in chunks:
                    ch_copy = dict(ch)
                    ch_copy["doc_id"] = doc_id
                    if ch_copy["chunk_id"] in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(ch_copy["chunk_id"])

                    emb = deterministic_embedding(
                        f"{ch_copy['title']} {ch_copy['heading_path']} {ch_copy['section']} {ch_copy['content']}"
                    )
                    existing_chunk = (
                        await session.execute(select(DocChunk).where(DocChunk.chunk_id == ch_copy["chunk_id"]))
                    ).scalar_one_or_none()
                    if existing_chunk:
                        existing_chunk.doc_id = doc_id
                        existing_chunk.content = ch_copy["content"]
                        existing_chunk.section = ch_copy["section"]
                        existing_chunk.heading_path = ch_copy["heading_path"]
                        existing_chunk.version = ch_copy["version"]
                        existing_chunk.superseded_by = ch_copy["superseded_by"]
                        existing_chunk.applies_to = ch_copy["applies_to"]
                        existing_chunk.embedding = emb
                    else:
                        payload = {k: v for k, v in ch_copy.items() if hasattr(DocChunk, k)}
                        payload["embedding"] = emb
                        session.add(DocChunk(**payload))

        await session.commit()
        print("Seeded societies, clinicians, patients, documents, and doc_chunks successfully.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", str(ROOT_DIR))
    asyncio.run(seed_all())