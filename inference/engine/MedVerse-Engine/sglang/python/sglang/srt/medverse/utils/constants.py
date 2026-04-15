"""Medical constants, section tags, and engine-wide limits."""

from typing import Final

# ── Engine limits ──────────────────────────────────────────────────────────────
MAX_BATCH_SIZE: Final[int] = 50          # inherited from Multiverse KV-cache constraint
MAX_DAG_DEPTH: Final[int] = 5           # default; configurable per request
MAX_PARALLEL_BRANCHES: Final[int] = 4

# ── KV cache ───────────────────────────────────────────────────────────────────
CACHE_KEY_PREFIX_TOKENS: Final[int] = 32  # tokens used for cache-key hashing

# ── Confidence thresholds ──────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.3
MIN_CONFIDENCE_TO_KEEP: Final[float] = 0.1

# ── Section header patterns (regex-ready) ─────────────────────────────────────
SECTION_PATTERNS: dict[str, list[str]] = {
    "HPI": [
        r"\bHPI\b",
        r"\bHistory of Present Illness\b",
        r"\bChief Complaint\b",
        r"\bCC\b",
        r"\bSubjective\b",
        r"\bPresenting Complaint\b",
    ],
    "VITALS": [
        r"\bVitals?\b",
        r"\bVital Signs?\b",
        r"\bVS\b",
        r"\bObjective\b",
    ],
    "LABS": [
        r"\bLabs?\b",
        r"\bLaboratory\b",
        r"\bLab(?:oratory)? Results?\b",
        r"\bBlood Work\b",
        r"\bCBC\b",
        r"\bBMP\b",
        r"\bCMP\b",
    ],
    "IMAGING": [
        r"\bImaging\b",
        r"\bRadiology\b",
        r"\bCXR\b",
        r"\bCT\b",
        r"\bMRI\b",
        r"\bX-?Ray\b",
        r"\bECG\b",
        r"\bEKG\b",
        r"\bEcho(?:cardiography)?\b",
        r"\bUltrasound\b",
    ],
    "PHYSICAL_EXAM": [
        r"\bPhysical Exam(?:ination)?\b",
        r"\bPE\b",
        r"\bExam\b",
        r"\bReview of Systems?\b",
        r"\bROS\b",
    ],
    "MEDICATIONS": [
        r"\bMedications?\b",
        r"\bMeds?\b",
        r"\bCurrent Medications?\b",
        r"\bHome Medications?\b",
        r"\bDrug(?:s)?\b",
    ],
    "DIFFERENTIAL_DIAGNOSIS": [
        r"\bDifferential(?:\s+Diagnosis)?\b",
        r"\bDDx\b",
        r"\bDiff(?:erential)?\b",
    ],
    "ASSESSMENT": [
        r"\bAssessment\b",
        r"\bImpression\b",
        r"\bDiagnosis\b",
        r"\bDx\b",
    ],
    "PLAN": [
        r"\bPlan\b",
        r"\bManagement\b",
        r"\bTreatment\b",
        r"\bDisposition\b",
        r"\bOrders?\b",
    ],
}

# Sections that are independent (parallelizable)
INDEPENDENT_SECTIONS: Final[frozenset[str]] = frozenset(
    ["VITALS", "LABS", "IMAGING", "PHYSICAL_EXAM", "MEDICATIONS"]
)

# Sections that depend on evidence (merge nodes)
EVIDENCE_DEPENDENT_SECTIONS: Final[frozenset[str]] = frozenset(
    ["DIFFERENTIAL_DIAGNOSIS", "ASSESSMENT"]
)

# Terminal section
TERMINAL_SECTIONS: Final[frozenset[str]] = frozenset(["PLAN"])
