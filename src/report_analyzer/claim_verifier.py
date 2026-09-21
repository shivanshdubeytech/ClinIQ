"""Stage F: Second-Pass Claim Verifier Against Retrieved Trusted Sources.

Performs automated fact-checking on every sentence in the generated explanation:
  1. Checks each claim against the retrieved source text.
  2. Removes or flags any unsupported or hallucinated assertions.
  3. Enforces Hard Rule 7: Detects and strips any prohibited diagnostic claims
     or dosage modification instructions.
  4. Computes a groundedness score (0.0 to 1.0) for the explanation.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from .explainer import ItemExplanation, ReportExplanationResult, UNSUPPORTED_FALLBACK_TEXT
from .knowledge_base import TrustedSource, retrieve_trusted_source


# Prohibited phrases according to Rule 7 (No diagnosis, no dose changes, no treatment advice)
DIAGNOSTIC_PATTERNS = [
    r"\b(you have|you are suffering from|you are diagnosed with|this proves you have|this confirms you have)\b",
    r"\b(my diagnosis is|the diagnosis is confirmed|you definitely have)\b",
]

TREATMENT_ADVICE_PATTERNS = [
    r"\b(increase your dose|decrease your dose|double your dose|cut your dose)\b",
    r"\b(stop taking|start taking|take\s+[0-9]+\s*mg\s+instead|switch to)\b",
    r"\b(you should self-medicate|prescribe yourself|adjust your medication)\b",
]

DIAGNOSTIC_REGEX = re.compile("|".join(DIAGNOSTIC_PATTERNS), re.IGNORECASE)
TREATMENT_REGEX = re.compile("|".join(TREATMENT_ADVICE_PATTERNS), re.IGNORECASE)


@dataclass
class VerifiedClaim:
    """Represents a single atomic claim evaluated against source text."""
    claim_text: str
    is_supported: bool
    confidence: float
    violates_safety_rule: bool = False
    safety_reason: Optional[str] = None
    supporting_source_segment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "is_supported": self.is_supported,
            "confidence": round(self.confidence, 3),
            "violates_safety_rule": self.violates_safety_rule,
            "safety_reason": self.safety_reason,
            "supporting_source_segment": self.supporting_source_segment,
        }


@dataclass
class SecondPassVerificationResult:
    """Encapsulates the second-pass review of an explanation."""
    original_text: str
    verified_text: str
    claims: List[VerifiedClaim] = field(default_factory=list)
    groundedness_ratio: float = 1.0
    has_unsupported_claims: bool = False
    has_safety_violations: bool = False
    unsupported_claims: List[str] = field(default_factory=list)
    removed_statements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "verified_text": self.verified_text,
            "groundedness_ratio": round(self.groundedness_ratio, 3),
            "has_unsupported_claims": self.has_unsupported_claims,
            "has_safety_violations": self.has_safety_violations,
            "claims_count": len(self.claims),
            "unsupported_claims": self.unsupported_claims,
            "removed_statements": self.removed_statements,
        }


def split_into_claims(text: str) -> List[str]:
    """Splits a paragraph into distinct sentence-level claims for verification."""
    # Split by periods, exclamation marks, or question marks followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for s in raw_sentences:
        cleaned = s.strip()
        # Filter out citations and short transitional phrases
        if len(cleaned) > 15 and not cleaned.startswith("According to") and not cleaned.startswith("Source:"):
            claims.append(cleaned)
    return claims


def _calculate_claim_support(claim: str, source_text: str) -> Tuple[bool, float, Optional[str]]:
    """Evaluates whether a claim is factual and grounded in the source text."""
    claim_norm = re.sub(r"[^a-zA-Z0-9\s]", " ", claim.lower())
    source_norm = re.sub(r"[^a-zA-Z0-9\s]", " ", source_text.lower())

    claim_words = [w for w in claim_norm.split() if len(w) > 3]
    if not claim_words:
        return True, 1.0, None

    # Stopwords to ignore
    common_words = {
        "this", "that", "with", "from", "your", "have", "been", "were", "what",
        "which", "there", "their", "about", "according", "these", "those",
        "level", "levels", "range", "measured", "standard", "reference",
    }
    content_words = [w for w in claim_words if w not in common_words]
    if not content_words:
        return True, 0.9, None

    # Count how many meaningful content words exist in the source document
    matched_words = [w for w in content_words if w in source_norm]
    support_ratio = len(matched_words) / len(content_words)

    # Threshold for factual support
    is_supported = support_ratio >= 0.45

    # Find the most matching sentence in the source for citation evidence
    best_segment = None
    if is_supported:
        source_sentences = re.split(r"(?<=[.!?])\s+", source_text)
        best_overlap = 0
        for sent in source_sentences:
            overlap = sum(1 for w in content_words if w in sent.lower())
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment = sent.strip()

    return is_supported, support_ratio, best_segment


def verify_claim(claim: str, source_text: str) -> VerifiedClaim:
    """Verifies an individual claim against safety rules and retrieved source text."""
    # 1. Hard Rule 7 check: Prohibited diagnostic statements
    if DIAGNOSTIC_REGEX.search(claim):
        return VerifiedClaim(
            claim_text=claim,
            is_supported=False,
            confidence=0.0,
            violates_safety_rule=True,
            safety_reason="Prohibited diagnostic statement detected (Rule 7).",
        )

    # 2. Hard Rule 7 check: Prohibited treatment / dose modification advice
    if TREATMENT_REGEX.search(claim):
        return VerifiedClaim(
            claim_text=claim,
            is_supported=False,
            confidence=0.0,
            violates_safety_rule=True,
            safety_reason="Prohibited treatment/dosage modification advice detected (Rule 7).",
        )

    # 3. Grounding check against source text
    is_supported, ratio, best_seg = _calculate_claim_support(claim, source_text)

    return VerifiedClaim(
        claim_text=claim,
        is_supported=is_supported,
        confidence=ratio,
        violates_safety_rule=False,
        supporting_source_segment=best_seg,
    )


def verify_explanation_second_pass(
    explanation: ItemExplanation,
    source: Optional[TrustedSource] = None,
) -> SecondPassVerificationResult:
    """Runs a second pass on an ItemExplanation to filter or flag unsupported claims.

    Args:
        explanation: The ItemExplanation generated in Stage E.
        source: The trusted source object (looked up if not provided).

    Returns:
        SecondPassVerificationResult with sanitized text and verification logs.
    """
    if not explanation.is_retrieved or explanation.explanation_text == UNSUPPORTED_FALLBACK_TEXT:
        return SecondPassVerificationResult(
            original_text=explanation.explanation_text,
            verified_text=explanation.explanation_text,
            groundedness_ratio=1.0,
            has_unsupported_claims=False,
            has_safety_violations=False,
        )

    if not source:
        source = retrieve_trusted_source(explanation.item_name)

    # Aggregate source content: trusted literature + verified structured finding + real drug database
    drug_context = ""
    try:
        from .drug_verifier import REAL_DRUG_DATABASE
        cleaned_key = explanation.item_name.lower().strip()
        if cleaned_key in REAL_DRUG_DATABASE:
            d_data = REAL_DRUG_DATABASE[cleaned_key]
            drug_context = f"{d_data['class']} {' '.join(d_data['brands'])} {' '.join(d_data['indications'])}"
    except Exception:
        pass

    report_finding_context = (
        f"{explanation.item_name} {explanation.measured_or_ordered} "
        "is within below above reference range normal low high limit"
    )
    source_corpus = (
        f"{report_finding_context} {drug_context} {source.summary_content} {' '.join(source.key_points)}"
        if source else ""
    )

    claims = split_into_claims(explanation.explanation_text)
    evaluated_claims: List[VerifiedClaim] = []
    verified_sentences: List[str] = []
    unsupported_claims: List[str] = []
    removed_statements: List[str] = []

    has_safety_violations = False

    for c in claims:
        v_claim = verify_claim(c, source_corpus)
        evaluated_claims.append(v_claim)

        if v_claim.violates_safety_rule:
            has_safety_violations = True
            removed_statements.append(f"{c} (Reason: {v_claim.safety_reason})")
            continue

        if v_claim.is_supported:
            verified_sentences.append(c)
        else:
            unsupported_claims.append(c)
            # Flag unsupported assertion instead of silently presenting it
            verified_sentences.append(f"[⚠️ Unverified Claim Flagged: '{c}']")

    total_claims = len(evaluated_claims)
    supported_count = sum(1 for c in evaluated_claims if c.is_supported and not c.violates_safety_rule)
    ratio = (supported_count / total_claims) if total_claims > 0 else 1.0

    # Reconstruct sanitized verified text
    sanitized_text = " ".join(verified_sentences)

    return SecondPassVerificationResult(
        original_text=explanation.explanation_text,
        verified_text=sanitized_text,
        claims=evaluated_claims,
        groundedness_ratio=ratio,
        has_unsupported_claims=len(unsupported_claims) > 0,
        has_safety_violations=has_safety_violations,
        unsupported_claims=unsupported_claims,
        removed_statements=removed_statements,
    )


def apply_second_pass_verification(
    report_result: ReportExplanationResult,
) -> ReportExplanationResult:
    """Applies second-pass verification to all explanations in a document result."""
    sanitized_explanations = []

    for exp in report_result.explanations:
        v_res = verify_explanation_second_pass(exp)

        # Update explanation with verified sanitized text
        verified_exp = ItemExplanation(
            item_name=exp.item_name,
            item_type=exp.item_type,
            measured_or_ordered=exp.measured_or_ordered,
            explanation_text=v_res.verified_text,
            source_name=exp.source_name,
            source_url=exp.source_url,
            is_retrieved=exp.is_retrieved,
            suggested_doctor_questions=exp.suggested_doctor_questions,
            critical_alert=exp.critical_alert,
        )
        sanitized_explanations.append(verified_exp)

    return ReportExplanationResult(
        document_summary=report_result.document_summary,
        explanations=sanitized_explanations,
        unsupported_items=list(report_result.unsupported_items),
        disclaimer=report_result.disclaimer,
    )
