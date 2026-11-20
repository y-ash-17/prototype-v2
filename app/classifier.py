import asyncio
import logging
from enum import Enum
from typing import Any

from app.safety import SafetyAssessment, assess_safety

logger = logging.getLogger("concierge.safety")


class Service(str, Enum):
    DOCTOR = "DOCTOR"
    LAB = "LAB"


class Intent(str, Enum):
    BOOK_SERVICE = "BOOK_SERVICE"
    LIST_SERVICES = "LIST_SERVICES"
    LOG_SYMPTOM = "LOG_SYMPTOM"
    CLINIC_INFO = "CLINIC_INFO"
    MEDICAL_ADVICE = "MEDICAL_ADVICE"
    EMERGENCY = "EMERGENCY"
    SMALL_TALK = "SMALL_TALK"
    UNKNOWN = "UNKNOWN"


class SafetyVerdict(str, Enum):
    EMERGENCY = "EMERGENCY"
    MEDICAL_ADVICE = "MEDICAL_ADVICE"
    SAFE = "SAFE"


_MEDICAL_ADVICE_PATTERNS = [
    "diagnose",
    "diagnosis",
    "what do i have",
    "what disease",
    "prescribe",
    "prescription",
    "which medicine",
    "what medicine",
    "what tablet",
    "what antibiotic",
    "dosage",
    "dose",
    "how many mg",
    "how much paracetamol",
    "how much ibuprofen",
    "dawai",
    "goli",
    "kitni dose",
]


def _is_medical_advice_query(msg_lower: str) -> bool:
    return any(pat in msg_lower for pat in _MEDICAL_ADVICE_PATTERNS)


def _infer_intent(message: str, verdict: SafetyVerdict) -> Intent:
    if verdict == SafetyVerdict.EMERGENCY:
        return Intent.EMERGENCY
    if verdict == SafetyVerdict.MEDICAL_ADVICE:
        return Intent.MEDICAL_ADVICE

    text = (message or "").strip()
    lower = text.lower()

    if any(w in lower for w in ("hello", "hi ", "hi!", "hey", "namaste", "good morning", "good evening", "thanks", "thank you")):
        if len(lower.split()) <= 4 and not any(
            k in lower for k in ("book", "test", "doctor", "slot", "pain", "fever", "symptom", "hours", "wellness")
        ):
            return Intent.SMALL_TALK

    if any(w in lower for w in ("my bookings", "list bookings", "upcoming bookings", "past bookings", "show bookings", "cancel")):
        return Intent.LIST_SERVICES

    if any(w in lower for w in ("book", "appointment", "consult", "lab test", "blood test", "fasting", "slot")):
        if any(ch.isdigit() for ch in lower) or "cardiology" in lower or "doctor" in lower or "lab" in lower or "khanna" in lower or "nair" in lower:
            return Intent.BOOK_SERVICE

    if any(w in lower for w in ("log symptom", "symptom", "severity", "feeling feverish", "mild fever", "headache", "bukhar")):
        if any(k in lower for k in ("severity", "level", "log", "1", "2", "3", "4", "5", "mild", "moderate", "severe")):
            return Intent.LOG_SYMPTOM

    if any(w in lower for w in ("wellness", "hours", "open", "sunday", "coverage", "plan", "thyroid", "lipid", "fasting", "hospital", "phone", "emergency", "policy", "annex", "kab khulta", "khulta hai")):
        return Intent.CLINIC_INFO

    return Intent.UNKNOWN


class SafetyAndIntentClassifier:
    """Safety-first triage and intent classifier with fail-closed error/timeout handling."""

    def __init__(
        self,
        llm: Any = None,
        classifier_fn: Any = None,
        timeout_seconds: float = 2.0,
        **kwargs: Any,
    ):
        self.llm = llm
        self.classifier_fn = classifier_fn
        self.timeout_seconds = timeout_seconds
        self.kwargs = kwargs
        self.failure_count = 0

    def _record_failure(self, reason: str, message: str) -> SafetyAssessment:
        self.failure_count += 1
        logger.error(
            "safety_classifier_failure count=%d reason=%s message_len=%d",
            self.failure_count,
            reason,
            len(message or ""),
        )
        return SafetyAssessment(level=1, reason="Safety classifier failed", call_112=True, sla_minutes=5)

    async def classify_safety(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> SafetyAssessment:
        _ = context
        text = message or ""
        try:
            if callable(self.classifier_fn):
                res = self.classifier_fn(text, context or {})
                if hasattr(res, "__await__"):
                    res = await asyncio.wait_for(res, timeout=self.timeout_seconds)
                if isinstance(res, SafetyAssessment):
                    return res
                if isinstance(res, dict):
                    lvl = res.get("level", res.get("safety_level"))
                    return SafetyAssessment(
                        level=lvl,
                        reason=str(res.get("reason", "Classifier assessment")),
                        call_112=bool(res.get("call_112", lvl == 1)),
                        sla_minutes=res.get("sla_minutes", 5 if lvl == 1 else (30 if lvl == 2 else (240 if lvl == 3 else None))),
                    )

            assessment = assess_safety(text, classifier_fn=None)
            if assessment.level is None and _is_medical_advice_query(text.lower()):
                return SafetyAssessment(level=None, call_112=False, sla_minutes=None)
            return assessment
        except asyncio.TimeoutError:
            return self._record_failure("timeout", text)
        except Exception as exc:
            return self._record_failure(type(exc).__name__, text)

    async def classify(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = message or ""
        failed = False
        try:
            safety = await asyncio.wait_for(
                self.classify_safety(text, context=context),
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            failed = True
            safety = self._record_failure(type(exc).__name__, text)

        if safety.level == 1:
            verdict = SafetyVerdict.EMERGENCY
        elif _is_medical_advice_query(text.lower()):
            verdict = SafetyVerdict.MEDICAL_ADVICE
        else:
            verdict = SafetyVerdict.SAFE

        intent = _infer_intent(text, verdict)
        return {
            "verdict": verdict.value,
            "safety_verdict": verdict.value,
            "intent": intent.value,
            "safety_level": safety.level,
            "call_112": safety.call_112,
            "sla_minutes": safety.sla_minutes,
            "classifier_failed": failed,
        }