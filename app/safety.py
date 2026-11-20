import re
from dataclasses import dataclass


@dataclass
class SafetyAssessment:
    level: int | None
    reason: str
    call_112: bool
    sla_minutes: int | None


LEVEL_1_PATTERNS = [
    r"\bsaans\s+(?:nahi|nahin|nhi)\s+aa\s+rahi\b",
    r"\bchest\s*pain\b",
    r"\bchest\s*pressure\b",
    r"\bdifficulty\s*breathing\b",
    r"\bshortness\s*of\s*breath\b",
    r"\bcan\s*'?t\s*breathe\b",
    r"\bcannot\s*breathe\b",
    r"\bunconscious\b",
    r"\bcollapsed\b",
    r"\bnot\s*responding\b",
    r"\bsevere\s*bleeding\b",
    r"\bheavy\s*bleeding\b",
    r"\bweakness\s*(of|in)?\s*(the\s*)?(face|arm|leg)\b",
    r"\bslurred\s*speech\b",
    r"\bseizure\b",
    r"\bsuicid(al|e)\b",
    r"\bend(ing)?\s*(my|their|his|her)?\s*life\b",
    r"\bwant\s*to\s*die\b",
]

LEVEL_2_PATTERNS = [
    r"\binfant\b.*\bfever\b",
    r"\bhigh\s*fever\b.*\b(baby|infant|child)\b",
    r"\bfracture\b",
    r"\bbroken\s*(bone|arm|leg|wrist|ankle)\b",
    r"\bfall\b.*\b(pain|swelling|fracture)\b",
    r"\bvomiting\b.*(stop|non\s*-?stop|persist|blood)",
    r"\bdeep\s*cut\b",
    r"\bstitches\b",
]

LEVEL_3_PATTERNS = [
    r"\bnurse\s*(to\s*)?(look|check|call|visit)\b",
    r"\bmild\s*fever\b",
    r"\brash\b",
    r"\bdizziness\b",
    r"\broutine\s*check\b",
]


def assess_safety(message: str, classifier_fn=None) -> SafetyAssessment:
    text = (message or "").strip()
    lowered = text.lower()

    if classifier_fn is not None:
        try:
            res = classifier_fn(text)
            if isinstance(res, SafetyAssessment):
                return res
            if isinstance(res, dict) and res.get("level") in (1, 2, 3):
                lvl = int(res["level"])
                sla = {1: 5, 2: 30, 3: 240}[lvl]
                return SafetyAssessment(
                    level=lvl,
                    reason=str(res.get("reason", "Classifier escalation")),
                    call_112=(lvl == 1),
                    sla_minutes=sla,
                )
        except Exception:
            return SafetyAssessment(
                level=1,
                reason="Safety classifier error/timeout; defaulting to Level 1 emergency escalation.",
                call_112=True,
                sla_minutes=5,
            )

    for pat in LEVEL_1_PATTERNS:
        if re.search(pat, lowered):
            return SafetyAssessment(level=1, reason=f"Matched critical symptom pattern: {pat}", call_112=True, sla_minutes=5)

    for pat in LEVEL_2_PATTERNS:
        if re.search(pat, lowered):
            return SafetyAssessment(level=2, reason=f"Matched urgent symptom pattern: {pat}", call_112=False, sla_minutes=30)

    for pat in LEVEL_3_PATTERNS:
        if re.search(pat, lowered):
            return SafetyAssessment(level=3, reason=f"Matched routine nurse review pattern: {pat}", call_112=False, sla_minutes=240)

    return SafetyAssessment(level=None, reason="No emergency or nurse escalation indicators detected.", call_112=False, sla_minutes=None)
