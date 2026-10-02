"""Stage G: Standardized Output Labeling & End-to-End Report Analyzer Pipeline.

Coordinates Stages A through F into a unified pipeline and assigns mandatory output labels:
  - "clearly read" (for high-confidence, clean extractions)
  - "low confidence - please verify" (for OCR-degraded, noisy, or unverified items)
  - "see a doctor soon" (for critical panic thresholds requiring prompt medical attention)
Always appends a short medical disclaimer.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .claim_verifier import apply_second_pass_verification
from .comparator import EvaluatedLabTest, compare_lab_report
from .drug_verifier import VerifiedDrug, verify_prescriptions
from .explainer import (
    ItemExplanation,
    ReportExplanationResult,
    STANDARD_REPORT_DISCLAIMER,
    explain_analyzed_document,
)
from .extractor import ExtractionResult, extract_medical_document
from .structurer import (
    LabTestItem,
    PrescriptionItem,
    StructuredDocument,
    structure_medical_document,
)

LABEL_CLEARLY_READ = "clearly read"
LABEL_LOW_CONFIDENCE = "low confidence - please verify"
LABEL_SEE_DOCTOR_SOON = "see a doctor soon"


@dataclass
class LabeledReportItem:
    """Represents an individual clinical item with its standardized user-facing label."""
    item_name: str
    item_type: str  # "lab_test", "prescription"
    measured_or_ordered: str
    status: str  # "LOW", "NORMAL", "HIGH", "VERIFIED", "UNVERIFIED", "UNKNOWN_RANGE"
    label: str  # "clearly read", "low confidence - please verify", "see a doctor soon"
    badge_style: str  # "success", "warning", "danger"
    explanation: str
    source_name: Optional[str]
    source_url: Optional[str]
    confidence: float
    suggested_doctor_questions: List[str] = field(default_factory=list)
    critical_alert: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_name": self.item_name,
            "item_type": self.item_type,
            "measured_or_ordered": self.measured_or_ordered,
            "status": self.status,
            "label": self.label,
            "badge_style": self.badge_style,
            "explanation": self.explanation,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "confidence": round(self.confidence, 3),
            "suggested_doctor_questions": self.suggested_doctor_questions,
            "critical_alert": self.critical_alert,
        }


@dataclass
class FinalAnalysisResult:
    """Complete, verified report analysis response with standardized labels and confirmation tables."""
    is_success: bool
    document_type: str
    items: List[LabeledReportItem] = field(default_factory=list)
    overall_label: str = LABEL_CLEARLY_READ
    overall_confidence: float = 1.0
    has_critical_findings: bool = False
    critical_alert: Optional[str] = None
    user_confirmation_table: str = ""
    summary_markdown: str = ""
    disclaimer: str = STANDARD_REPORT_DISCLAIMER
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_success": self.is_success,
            "document_type": self.document_type,
            "items": [it.to_dict() for it in self.items],
            "overall_label": self.overall_label,
            "overall_confidence": round(self.overall_confidence, 3),
            "has_critical_findings": self.has_critical_findings,
            "critical_alert": self.critical_alert,
            "user_confirmation_table": self.user_confirmation_table,
            "summary_markdown": self.summary_markdown,
            "disclaimer": self.disclaimer,
            "error": self.error,
        }


def assign_item_label(
    confidence: float,
    is_critical: bool,
    is_ambiguous: bool = False,
    is_verified_drug: bool = True,
    has_verification_flags: bool = False,
    status: Optional[str] = None,
) -> Tuple[str, str]:
    """Assigns the exact required label: 'see a doctor soon', 'low confidence - please verify', or 'clearly read'.

    Returns:
        Tuple of (label_text, badge_style)
    """
    # 1. Critical panic findings always take precedence
    if is_critical:
        return LABEL_SEE_DOCTOR_SOON, "danger"

    # 2. Low confidence, ambiguous formatting, unverified medications, unit mismatch, or second-pass verification flags
    if (
        confidence < 0.85
        or is_ambiguous
        or not is_verified_drug
        or has_verification_flags
        or status == "UNVERIFIED_UNIT"
    ):
        return LABEL_LOW_CONFIDENCE, "warning"

    # 3. Clean, high confidence read
    return LABEL_CLEARLY_READ, "success"


def analyze_medical_document(
    file_path: Union[str, Path],
    auto_delete_temp: bool = False,
    use_llm: bool = False,
) -> FinalAnalysisResult:
    """Complete end-to-end medical document analyzer implementing Stages A to G.

    Args:
        file_path: Path to the uploaded medical PDF or image.
        auto_delete_temp: When True, safely removes temporary upload after extraction (Rule 8).
        use_llm: Whether to invoke LLM for strictly grounded plain-language explanations.

    Returns:
        FinalAnalysisResult with structured items, standardized labels, and citations.
    """
    # ==========================================
    # Stage A: Extract
    # ==========================================
    extraction = extract_medical_document(file_path, auto_delete_temp=auto_delete_temp)
    if not extraction.is_success:
        return FinalAnalysisResult(
            is_success=False,
            document_type=extraction.document_type,
            overall_label=LABEL_LOW_CONFIDENCE,
            overall_confidence=0.0,
            error=extraction.error or "Failed to extract text from document.",
        )

    # ==========================================
    # Stage B: Structure into Tables
    # ==========================================
    doc_structure = structure_medical_document(extraction)
    confirmation_table = doc_structure.to_markdown_table()

    # ==========================================
    # Stage C: Compare with CODE (Lab Tests)
    # ==========================================
    evaluated_tests = compare_lab_report(doc_structure)

    # ==========================================
    # Stage D: Verify Drugs (Prescriptions)
    # ==========================================
    verified_drugs = verify_prescriptions(doc_structure)

    # ==========================================
    # Stage E: Explain over Trusted Sources
    # ==========================================
    raw_explanation = explain_analyzed_document(
        lab_tests=evaluated_tests,
        verified_drugs=verified_drugs,
        prescriptions=doc_structure.prescriptions,
        use_llm=use_llm,
    )

    # ==========================================
    # Stage F: Verify Claims (Second Pass)
    # ==========================================
    verified_explanation = apply_second_pass_verification(raw_explanation)

    # ==========================================
    # Stage G: Output Labels & Final Assembly
    # ==========================================
    labeled_items: List[LabeledReportItem] = []
    has_critical = False
    critical_alerts = []

    # Map Lab Tests to LabeledReportItems
    for i, test in enumerate(evaluated_tests):
        exp = (
            verified_explanation.explanations[i]
            if i < len(verified_explanation.explanations)
            else None
        )

        label, badge = assign_item_label(
            confidence=test.confidence,
            is_critical=test.is_critical,
            has_verification_flags=bool(exp.verification_flags) if exp else False,
            status=test.status,
        )

        if test.is_critical:
            has_critical = True
            if test.critical_alert:
                critical_alerts.append(test.critical_alert)

        labeled_items.append(
            LabeledReportItem(
                item_name=test.test_name,
                item_type="lab_test",
                measured_or_ordered=f"{test.value} {test.unit} ({test.status})",
                status=test.status,
                label=label,
                badge_style=badge,
                explanation=exp.explanation_text if exp else test.interpretation,
                source_name=exp.source_name if exp else None,
                source_url=exp.source_url if exp else None,
                confidence=test.confidence,
                suggested_doctor_questions=exp.suggested_doctor_questions if exp else [],
                critical_alert=test.critical_alert,
            )
        )

    # Map Prescriptions to LabeledReportItems
    offset = len(evaluated_tests)
    for j, drug in enumerate(verified_drugs):
        idx = offset + j
        exp = (
            verified_explanation.explanations[idx]
            if idx < len(verified_explanation.explanations)
            else None
        )
        rx_item = (
            doc_structure.prescriptions[j]
            if j < len(doc_structure.prescriptions)
            else None
        )

        label, badge = assign_item_label(
            confidence=rx_item.confidence if rx_item else 1.0,
            is_critical=False,
            is_ambiguous=rx_item.is_ambiguous if rx_item else False,
            is_verified_drug=drug.is_verified,
            has_verification_flags=bool(exp.verification_flags) if exp else False,
        )

        dosage_desc = (
            f"{rx_item.dose} {rx_item.frequency}" if rx_item else "Prescribed"
        )
        status_str = "VERIFIED" if drug.is_verified else "UNVERIFIED"

        labeled_items.append(
            LabeledReportItem(
                item_name=drug.matched_generic or drug.query_name,
                item_type="prescription",
                measured_or_ordered=dosage_desc,
                status=status_str,
                label=label,
                badge_style=badge,
                explanation=exp.explanation_text if exp else "Prescription verified.",
                source_name=exp.source_name if exp else None,
                source_url=exp.source_url if exp else None,
                confidence=rx_item.confidence if rx_item else 1.0,
                suggested_doctor_questions=exp.suggested_doctor_questions if exp else [],
                critical_alert=drug.warning,
            )
        )

    # Overall document label
    if has_critical:
        overall_doc_label = LABEL_SEE_DOCTOR_SOON
    elif (
        any(it.label == LABEL_LOW_CONFIDENCE for it in labeled_items)
        or extraction.is_blurry_or_low_quality
        or bool(verified_explanation.verification_flags)
    ):
        overall_doc_label = LABEL_LOW_CONFIDENCE
    else:
        overall_doc_label = LABEL_CLEARLY_READ

    crit_summary = (
        " | ".join(critical_alerts) if critical_alerts else None
    )

    return FinalAnalysisResult(
        is_success=True,
        document_type=doc_structure.document_type,
        items=labeled_items,
        overall_label=overall_doc_label,
        overall_confidence=extraction.overall_confidence,
        has_critical_findings=has_critical,
        critical_alert=crit_summary,
        user_confirmation_table=confirmation_table,
        summary_markdown=verified_explanation.to_markdown(),
        disclaimer=STANDARD_REPORT_DISCLAIMER,
    )
