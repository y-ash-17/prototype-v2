# MicroHeal Care Concierge

A society-scoped healthcare concierge prototype built with **LangGraph**, **LangChain**, **PostgreSQL 16 + pgvector**, **SQLAlchemy 2.x**, **Alembic**, and **FastAPI** (SSE streaming + single-file HTML frontend).

## Quickstart from a Clean Clone

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec app alembic upgrade head
docker compose exec app python -m scripts.seed_and_ingest

Open http://localhost:8000 in your browser, or verify health:

Bash
curl http://localhost:8000/healthz
Running Tests & Offline Evaluation (No API Key Required)
Bash
# Run offline evaluation suite (replays committed model outputs + local dense retrieval)
docker compose exec app make eval

# Run integration & concurrency tests against Postgres
docker compose exec app make test
Architecture Overview
Safety-First LangGraph (app/graph.py): START -> safety_node runs before any routing, retrieval, or tool node. EMERGENCY writes an escalations row via escalate_emergency and halts; MEDICAL_ADVICE refuses diagnosis/prescriptions/dosages and offers a consultation; only SAFE reaches router_node and tool_node.

Database-Enforced Invariants (alembic/versions/001_initial_schema.py):

Partial unique index uq_clinician_active_slot on (clinician_id, scheduled_for) WHERE status IN ('CONFIRMED', 'PENDING') prevents double-booking under concurrency.

UNIQUE constraint on service_requests.request_id, symptom_logs.request_id, and escalations.request_id ensures idempotent retries.

CHECK (severity >= 1 AND severity <= 5) enforces valid symptom severity at the database layer.

Centralized Society & Version Scoping (app/scoping.py): Every booking and document query requires society_id and filters is_current = TRUE AND (applies_to = :society_id OR applies_to = 'all') directly in SQL WHERE.

Prompt Injection Defense (app/ingest.py & app/graph.py): Directives addressed to the assistant inside staff-notes-internal.md are stripped at ingest and retrieved chunks are wrapped as untrusted data.

What We Cut and Why
WhatsApp Cloud API webhook ingress (fixtures/webhook/): Explicitly marked optional in the brief; prioritized core DB concurrency, safety gating, society/version isolation, and offline evals within the 8-hour timebox.

Hybrid lexical + dense RRF retrieval: Kept retrieval strictly dense pgvector cosine search (<=>) per the Core spec so retrieval behavior and threshold tuning remain transparent and easy to inspect.

Human-in-the-loop booking approval interrupt: Focused on single-pass slot filling with Postgres checkpoint persistence (AsyncPostgresSaver) and added the Tool-Call Timeline and Page Reload Transcript Restoration (GET /v1/threads/{thread_id}) instead.
## Demo status

Run `docker compose up -d --build`, then visit http://localhost:8000. Use the sample patient selector and suggested questions to explore the prototype. This demo has no patient authentication or live nurse notification; do not use real patient data or rely on it for clinical care. WhatsApp integration is not implemented.

The remaining work includes authentication and patient/thread authorization, clinician and booking validation (including timezone-safe database uniqueness and request-ID ownership), clinically reviewed multilingual safety and escalation notification, full symptom-logging chat integration, and independent evaluation without cached expected outputs. The current document retrieval uses deterministic hash embeddings and keyword scoring rather than a multilingual semantic embedding model.
