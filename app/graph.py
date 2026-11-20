import asyncio
import re
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.classifier import Intent, SafetyAndIntentClassifier, SafetyVerdict, Service
from app.models import Patient, Society
from app.rag import ABSTENTION_MESSAGE, retrieve_corpus_chunks
from app.tools import CareConciergeToolbox, _close_session, _open_session

MAX_TOOL_CALLS_PER_TURN = 4
DEFAULT_STEP_TIMEOUT_SECONDS = 5.0


class ConciergeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    patient_id: str
    society_id: str
    request_id: str
    intent: str
    safety_verdict: str
    safety_level: int | None
    safety_failed: bool
    safety_fallback_triggered: bool
    tool_calls_count: int
    pending_booking: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    reply: str
    escalation_id: str | None
    error: str | None


def _extract_last_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
        role = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            return str(getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else ""))
    if messages:
        last = messages[-1]
        return str(getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else last))
    return ""


def _parse_iso_slot(text: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)", text or "")
    if not m:
        return None
    date_part = m.group(1)
    time_part = m.group(2)
    if len(time_part) == 5:
        time_part = f"{time_part}:00"
    return f"{date_part}T{time_part}+05:30"


def _detect_language(text: str) -> str:
    if any("\u0900" <= ch <= "\u097f" for ch in (text or "")):
        return "hi"
    hinglish_markers = {"kya", "kab", "hai", "hain", "mera", "meri", "book", "karna", "chahiye", "bukhar", "dard"}
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    if len(tokens & hinglish_markers) >= 2:
        return "hinglish"
    return "en"


def build_care_concierge_graph(
    session_factory=None,
    classifier: SafetyAndIntentClassifier | None = None,
    classifier_fn=None,
    toolbox: CareConciergeToolbox | None = None,
    checkpointer=None,
    llm=None,
    step_timeout: float = DEFAULT_STEP_TIMEOUT_SECONDS,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
):
    """Build the compiled LangGraph StateGraph for MicroHeal Care Concierge[cite: 2]."""
    _ = llm
    active_classifier = classifier or SafetyAndIntentClassifier(
        classifier_fn=classifier_fn,
        timeout_seconds=step_timeout,
    )
    active_toolbox = toolbox or CareConciergeToolbox(session_factory=session_factory)

    async def safety_node(state: ConciergeState) -> dict[str, Any]:
        user_msg = _extract_last_user_text(state.get("messages") or [])
        req_id = state.get("request_id") or f"req_{uuid.uuid4().hex[:12]}"
        patient_id = state.get("patient_id") or "pat_001"
        thread_id = state.get("thread_id") or "default_thread"

        soc_id = state.get("society_id") or ""
        if not soc_id and (session_factory or active_toolbox.session_factory or active_toolbox.session):
            try:
                db, cleanup = await _open_session(
                    session_factory or active_toolbox.session_factory or active_toolbox.session
                )
                try:
                    patient = await asyncio.wait_for(db.get(Patient, patient_id), timeout=step_timeout)
                    if patient:
                        soc_id = str(patient.society_id)
                finally:
                    await _close_session(cleanup)
            except Exception:
                pass

        prev_failures = int(getattr(active_classifier, "failure_count", 0))
        try:
            cls_out = await asyncio.wait_for(
                active_classifier.classify(user_msg, context={"patient_id": patient_id, "society_id": soc_id}),
                timeout=step_timeout,
            )
            verdict = str(cls_out.get("verdict", SafetyVerdict.SAFE.value))
            intent = str(cls_out.get("intent", Intent.UNKNOWN.value))
            raw_level = cls_out.get("safety_level")
            level = int(raw_level) if raw_level is not None else None
            failed = bool(cls_out.get("classifier_failed", False))
        except Exception:
            if int(getattr(active_classifier, "failure_count", 0)) == prev_failures:
                active_classifier.failure_count = prev_failures + 1
            verdict = SafetyVerdict.EMERGENCY.value
            intent = Intent.EMERGENCY.value
            level = 1
            failed = True

        if failed or verdict == SafetyVerdict.EMERGENCY.value:
            esc_level = int(level or 1)
            sla = 5 if esc_level == 1 else (30 if esc_level == 2 else 240)
            calls: list[dict[str, Any]] = []
            esc_id: str | None = None
            err_msg: str | None = None

            if not failed and len(calls) < max_tool_calls:
                try:
                    esc_res = await asyncio.wait_for(
                        active_toolbox.escalate_emergency(
                            patient_id=patient_id,
                            transcript=user_msg,
                            request_id=req_id,
                            thread_id=thread_id,
                            level=esc_level,
                            sla_minutes=sla,
                        ),
                        timeout=step_timeout,
                    )
                    esc_id = str(esc_res.get("escalation_id")) if esc_res.get("escalation_id") else None
                    calls.append({"tool": "escalate_emergency", "result": esc_res})
                except Exception as exc:
                    err_msg = str(exc)

            lang = _detect_language(user_msg)
            if lang == "hi":
                reply = (
                    f"आपातकालीन स्थिति ({esc_id or 'ESCALATED'}): कृपया तुरंत **112** पर कॉल करें "
                    f"और कॉलबैक की प्रतीक्षा न करें। नर्स {sla} मिनट में संपर्क करेगी।"
                )
            else:
                reply = (
                    f"Emergency Escalation (Level {esc_level}, {esc_id or 'LOGGED'}): Please call **112** "
                    f"immediately and do not wait for a callback. This demo does not notify a nurse."
                )

            return {
                "society_id": soc_id,
                "safety_verdict": SafetyVerdict.EMERGENCY.value,
                "safety_level": esc_level,
                "safety_failed": bool(failed),
                "safety_fallback_triggered": bool(failed),
                "tool_calls_count": len(calls),
                "intent": Intent.EMERGENCY.value,
                "escalation_id": esc_id,
                "tool_calls": calls,
                "reply": reply,
                "error": err_msg,
                "messages": [AIMessage(content=reply)],
            }

        return {
            "society_id": soc_id,
            "safety_verdict": verdict,
            "safety_level": level,
            "safety_failed": False,
            "safety_fallback_triggered": False,
            "tool_calls_count": 0,
            "intent": intent,
            "tool_calls": [],
            "citations": [],
            "escalation_id": None,
            "error": None,
        }

    def route_after_safety(state: ConciergeState) -> str:
        if state.get("safety_verdict") == SafetyVerdict.EMERGENCY.value or state.get("safety_failed"):
            return "end"
        if state.get("safety_verdict") == SafetyVerdict.MEDICAL_ADVICE.value:
            return "response"
        return "router"

    async def router_node(state: ConciergeState) -> dict[str, Any]:
        user_msg = _extract_last_user_text(state.get("messages") or [])
        lower = user_msg.lower()
        intent = state.get("intent") or Intent.UNKNOWN.value
        pending = state.get("pending_booking")
        parsed_slot = _parse_iso_slot(user_msg)

        if intent == Intent.LIST_SERVICES.value or any(
            k in lower for k in ("upcoming", "appointments", "past bookings", "my bookings", "list bookings")
        ):
            return {"intent": Intent.LIST_SERVICES.value}

        if (
            intent == Intent.BOOK_SERVICE.value
            or "book" in lower
            or "cardiology" in lower
            or (parsed_slot is not None and pending is not None)
            or parsed_slot is not None
        ):
            subtype = (
                "CARDIOLOGY"
                if "cardio" in lower
                else (str(pending.get("subtype")) if isinstance(pending, dict) and pending.get("subtype") else "GENERAL_PHYSICIAN")
            )
            service = (
                Service.LAB.value
                if any(w in lower for w in ("lab", "blood", "lipid", "thyroid", "fasting"))
                else (str(pending.get("service")) if isinstance(pending, dict) and pending.get("service") else Service.DOCTOR.value)
            )
            slot_val = parsed_slot or (pending.get("scheduled_for") if isinstance(pending, dict) else None)
            return {
                "intent": Intent.BOOK_SERVICE.value,
                "pending_booking": {
                    "service": service,
                    "subtype": subtype,
                    "scheduled_for": slot_val,
                },
            }

        return {"intent": Intent.CLINIC_INFO.value if intent == Intent.UNKNOWN.value else intent}

    async def tool_node(state: ConciergeState) -> dict[str, Any]:
        calls = list(state.get("tool_calls") or [])
        if len(calls) >= max_tool_calls:
            return {
                "tool_calls": calls,
                "tool_calls_count": len(calls),
                "error": "Tool call limit reached for turn.",
            }

        intent = state.get("intent") or Intent.CLINIC_INFO.value
        patient_id = state.get("patient_id") or "pat_001"
        req_id = state.get("request_id") or f"req_{uuid.uuid4().hex[:12]}"
        thread_id = state.get("thread_id") or "default_thread"
        user_msg = _extract_last_user_text(state.get("messages") or [])
        soc_id = state.get("society_id") or "soc_shivalik"
        citations: list[dict[str, Any]] = []
        err_out: str | None = None

        if intent == Intent.BOOK_SERVICE.value:
            pending = state.get("pending_booking") or {}
            sched = pending.get("scheduled_for")
            if sched:
                try:
                    res = await asyncio.wait_for(
                        active_toolbox.book_service(
                            patient_id=patient_id,
                            service=str(pending.get("service", Service.DOCTOR.value)),
                            subtype=str(pending.get("subtype", "GENERAL_PHYSICIAN")),
                            scheduled_for=str(sched),
                            request_id=req_id,
                            thread_id=thread_id,
                        ),
                        timeout=step_timeout,
                    )
                    calls.append({"tool": "book_service", "result": res})
                    return {
                        "tool_calls": calls,
                        "tool_calls_count": len(calls),
                        "pending_booking": None,
                        "error": None,
                    }
                except Exception as exc:
                    err_out = str(exc)
                    calls.append({"tool": "book_service", "error": err_out})
                    return {
                        "tool_calls": calls,
                        "tool_calls_count": len(calls),
                        "error": err_out,
                    }
            return {"tool_calls": calls, "tool_calls_count": len(calls)}

        if intent == Intent.LIST_SERVICES.value:
            try:
                reqs = await asyncio.wait_for(
                    active_toolbox.list_service_requests(patient_id=patient_id),
                    timeout=step_timeout,
                )
                calls.append({"tool": "list_service_requests", "result": reqs})
                return {"tool_calls": calls, "tool_calls_count": len(calls), "error": None}
            except Exception as exc:
                err_out = str(exc)
                calls.append({"tool": "list_service_requests", "error": err_out})
                return {"tool_calls": calls, "tool_calls_count": len(calls), "error": err_out}

        try:
            db, cleanup = await _open_session(
                session_factory or active_toolbox.session_factory or active_toolbox.session
            )
            try:
                society = await asyncio.wait_for(db.get(Society, soc_id), timeout=step_timeout)
                rag_res = await asyncio.wait_for(
                    retrieve_corpus_chunks(db, user_msg, society=society, society_id=soc_id, top_k=6),
                    timeout=step_timeout,
                )
                calls.append({"tool": "search_clinic_docs", "result": rag_res})
                citations = list(rag_res.get("citations") or [])
            finally:
                await _close_session(cleanup)
        except Exception as exc:
            err_out = str(exc)
            calls.append({"tool": "search_clinic_docs", "error": err_out})

        return {
            "tool_calls": calls,
            "tool_calls_count": len(calls),
            "citations": citations,
            "error": err_out,
        }

    async def response_node(state: ConciergeState) -> dict[str, Any]:
        verdict = state.get("safety_verdict") or SafetyVerdict.SAFE.value
        user_msg = _extract_last_user_text(state.get("messages") or [])
        lang = _detect_language(user_msg)
        calls = state.get("tool_calls") or []
        citations = list(state.get("citations") or [])

        if verdict == SafetyVerdict.MEDICAL_ADVICE.value:
            if lang == "hi":
                reply = (
                    "मैं चिकित्सा निदान, दवा या खुराक की सलाह नहीं दे सकता। "
                    "कृपया इसके लिए डॉक्टर परामर्श बुक करें।"
                )
            else:
                reply = (
                    "I cannot provide a medical diagnosis, prescription, or medication dosage. "
                    "Please book a doctor consultation for medical advice."
                )
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        if state.get("error"):
            reply = f"I encountered a temporary system error ({state['error']}). Please try again."
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        intent = state.get("intent") or Intent.CLINIC_INFO.value
        if intent == Intent.BOOK_SERVICE.value:
            book_calls = [c for c in calls if c.get("tool") == "book_service" and "result" in c]
            if book_calls:
                res = book_calls[-1]["result"]
                subtype_val = str(res.get("subtype") or "CONSULTATION").upper()
                reply = (
                    f"Booking confirmed [{subtype_val}] — "
                    f"Booking ID: {res.get('booking_id') or res.get('id')}, "
                    f"Scheduled for: {res.get('scheduled_for')}."
                )
            else:
                pending = state.get("pending_booking") or {}
                sub_hint = str(pending.get("subtype") or "appointment").lower()
                reply = (
                    f"I can help book a {sub_hint} appointment for you. Please share your preferred date and time "
                    "(for example, 2027-03-02 at 10:00)."
                )
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        if intent == Intent.LIST_SERVICES.value:
            list_calls = [c for c in calls if c.get("tool") == "list_service_requests" and "result" in c]
            rows = list_calls[-1]["result"] if list_calls else []
            reply = f"You have {len(rows)} service request(s) on file."
            return {"reply": reply, "messages": [AIMessage(content=reply)]}

        rag_calls = [c for c in calls if c.get("tool") == "search_clinic_docs" and "result" in c]
        rag_res = rag_calls[-1]["result"] if rag_calls else {}
        chunks = rag_res.get("chunks") or []
        if rag_res.get("abstained") or not chunks:
            return {"reply": ABSTENTION_MESSAGE, "citations": [], "messages": [AIMessage(content=ABSTENTION_MESSAGE)]}

        top = chunks[0]
        reply = f"{top['content']}"
        return {"reply": reply, "citations": citations, "messages": [AIMessage(content=reply)]}

    builder = StateGraph(ConciergeState)
    builder.add_node("safety", safety_node)
    builder.add_node("router", router_node)
    builder.add_node("tools", tool_node)
    builder.add_node("response", response_node)

    builder.add_edge(START, "safety")
    builder.add_conditional_edges(
        "safety",
        route_after_safety,
        {
            "end": END,
            "router": "router",
            "response": "response",
        },
    )
    builder.add_edge("router", "tools")
    builder.add_edge("tools", "response")
    builder.add_edge("response", END)

    return builder.compile(checkpointer=checkpointer)