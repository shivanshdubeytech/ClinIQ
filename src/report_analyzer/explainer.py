"""Stage E: Plain-Language Medical Explanations with Strict Grounding over Trusted Sources.

Uses retrieval (RAG) exclusively over authoritative sources (MedlinePlus, WHO, NIH, FDA).
Strict constraints:
  1. The explanation may ONLY use facts directly present in the retrieved source.
  2. Every explanation MUST explicitly cite its source name.
  3. If nothing is retrieved, outputs: "I don't have reliable information on this."
  4. Never invents facts, never provides diagnosis, dose changes, or treatment advice.
  5. Suggests questions for the patient to ask their physician.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional

from .comparator import EvaluatedLabTest
from .drug_verifier import VerifiedDrug
from .knowledge_base import TrustedSource, retrieve_trusted_source
from .structurer import PrescriptionItem, StructuredDocument

STANDARD_REPORT_DISCLAIMER = (
    "Educational Notice: This summary is generated from trusted clinical reference materials "
    "(MedlinePlus, NIH, WHO, FDA) for educational purposes only. It is not a medical diagnosis, "
    "does not evaluate individual health history, and does not alter prescriptions. "
    "Always consult your licensed physician or healthcare provider for medical decisions."
)

UNSUPPORTED_FALLBACK_TEXT = "I don't have reliable information on this."


@dataclass
class ItemExplanation:
    """Represents a plain-language explanation for a lab test or medication order."""
    item_name: str
    item_type: str  # "lab_test", "prescription", "unrecognized"
    measured_or_ordered: str
    explanation_text: str
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    is_retrieved: bool = False
    suggested_doctor_questions: List[str] = field(default_factory=list)
    critical_alert: Optional[str] = None
    verification_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_name": self.item_name,
            "item_type": self.item_type,
            "measured_or_ordered": self.measured_or_ordered,
            "explanation_text": self.explanation_text,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "is_retrieved": self.is_retrieved,
            "suggested_doctor_questions": self.suggested_doctor_questions,
            "critical_alert": self.critical_alert,
            "verification_flags": self.verification_flags,
        }


@dataclass
class ReportExplanationResult:
    """Encapsulates the complete grounded explanation result for a medical document."""
    document_summary: str
    explanations: List[ItemExplanation] = field(default_factory=list)
    unsupported_items: List[str] = field(default_factory=list)
    disclaimer: str = STANDARD_REPORT_DISCLAIMER
    verification_flags: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Renders an accessible plain-language Markdown summary for patients."""
        md = [f"## 📋 Plain-Language Summary\n\n{self.document_summary}\n"]

        if self.explanations:
            md.append("### 🔍 Verified Findings & Trusted Explanations\n")
            for exp in self.explanations:
                badge = "🔬 Lab Test" if exp.item_type == "lab_test" else "💊 Medication"
                md.append(f"#### {badge}: {exp.item_name} ({exp.measured_or_ordered})")

                if exp.critical_alert:
                    md.append(f"> 🚨 **{exp.critical_alert}**\n")

                md.append(f"{exp.explanation_text}\n")

                if exp.source_name:
                    source_link = (
                        f"[{exp.source_name}]({exp.source_url})"
                        if exp.source_url
                        else exp.source_name
                    )
                    md.append(f"**Source:** {source_link}\n")

                if exp.suggested_doctor_questions:
                    md.append("**Questions to ask your doctor:**")
                    for q in exp.suggested_doctor_questions:
                        md.append(f"- {q}")
                    md.append("")

        if self.unsupported_items:
            md.append("### ⚠️ Unverified Items")
            for item in self.unsupported_items:
                md.append(f"- **{item}**: {UNSUPPORTED_FALLBACK_TEXT}")
            md.append("")

        md.append(f"---\n*{self.disclaimer}*")
        return "\n".join(md)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_summary": self.document_summary,
            "explanations": [e.to_dict() for e in self.explanations],
            "unsupported_items": self.unsupported_items,
            "disclaimer": self.disclaimer,
            "verification_flags": self.verification_flags,
        }


def explain_lab_test(
    test: EvaluatedLabTest,
    use_llm: bool = False,
) -> ItemExplanation:
    """Generates an explanation for a single lab test strictly grounded in retrieved sources.

    If nothing is retrieved from trusted sources, returns:
    'I don't have reliable information on this.'
    """
    source = retrieve_trusted_source(test.test_name)

    if not source:
        return ItemExplanation(
            item_name=test.test_name,
            item_type="lab_test",
            measured_or_ordered=f"{test.value} {test.unit}".strip(),
            explanation_text=UNSUPPORTED_FALLBACK_TEXT,
            source_name=None,
            source_url=None,
            is_retrieved=False,
            critical_alert=test.critical_alert,
        )

    # Grounded synthesis strictly from retrieved source and measured test interpretation
    measured_str = f"{test.value} {test.unit} (Status: {test.status})"
    grounded_explanation = (
        f"{test.interpretation} According to {source.source_name}: {source.summary_content}"
    )

    return ItemExplanation(
        item_name=test.test_name,
        item_type="lab_test",
        measured_or_ordered=measured_str,
        explanation_text=grounded_explanation,
        source_name=source.source_name,
        source_url=source.source_url,
        is_retrieved=True,
        suggested_doctor_questions=list(source.suggested_doctor_questions),
        critical_alert=test.critical_alert,
    )


def explain_prescription(
    drug: VerifiedDrug,
    rx_item: Optional[PrescriptionItem] = None,
    use_llm: bool = False,
) -> ItemExplanation:
    """Generates an explanation for a medication order strictly grounded in retrieved sources.

    If nothing is retrieved from trusted sources, returns:
    'I don't have reliable information on this.'
    """
    lookup_name = drug.matched_generic or drug.query_name
    source = retrieve_trusted_source(lookup_name)

    dosage_str = f"{rx_item.dose} {rx_item.frequency}".strip() if rx_item else "Prescribed order"

    if not source or not drug.is_verified:
        return ItemExplanation(
            item_name=drug.query_name,
            item_type="prescription",
            measured_or_ordered=dosage_str,
            explanation_text=UNSUPPORTED_FALLBACK_TEXT,
            source_name=None,
            source_url=None,
            is_retrieved=False,
            critical_alert=drug.warning,
        )

    brand_note = (
        f" (commonly known as {', '.join(drug.brand_names[:2])})"
        if drug.brand_names
        else ""
    )

    grounded_explanation = (
        f"{drug.matched_generic}{brand_note} belongs to the class of {drug.drug_class}. "
        f"According to {source.source_name}: {source.summary_content}"
    )

    return ItemExplanation(
        item_name=drug.matched_generic or drug.query_name,
        item_type="prescription",
        measured_or_ordered=dosage_str,
        explanation_text=grounded_explanation,
        source_name=source.source_name,
        source_url=source.source_url,
        is_retrieved=True,
        suggested_doctor_questions=list(source.suggested_doctor_questions),
        critical_alert=drug.warning,
    )


def explain_analyzed_document(
    lab_tests: Optional[List[EvaluatedLabTest]] = None,
    verified_drugs: Optional[List[VerifiedDrug]] = None,
    prescriptions: Optional[List[PrescriptionItem]] = None,
    use_llm: bool = False,
) -> ReportExplanationResult:
    """Orchestrates comprehensive, grounded explanations for all parsed medical items.

    Args:
        lab_tests: List of EvaluatedLabTest items from Stage C.
        verified_drugs: List of VerifiedDrug items from Stage D.
        prescriptions: Optional list of raw PrescriptionItem items from Stage B.
        use_llm: Flag indicating whether to perform LLM re-phrasing (strictly grounded).

    Returns:
        ReportExplanationResult with Markdown rendering and source citations.
    """
    lab_tests = lab_tests or []
    verified_drugs = verified_drugs or []
    prescriptions = prescriptions or []

    explanations: List[ItemExplanation] = []
    unsupported_items: List[str] = []

    # 1. Process Lab Tests
    for test in lab_tests:
        exp = explain_lab_test(test, use_llm=use_llm)
        if exp.is_retrieved:
            explanations.append(exp)
        else:
            unsupported_items.append(test.test_name)

    # 2. Process Prescriptions
    for i, drug in enumerate(verified_drugs):
        rx_item = prescriptions[i] if i < len(prescriptions) else None
        exp = explain_prescription(drug, rx_item=rx_item, use_llm=use_llm)
        if exp.is_retrieved:
            explanations.append(exp)
        else:
            unsupported_items.append(drug.query_name)

    # 3. Overall document summary narrative
    total_items = len(lab_tests) + len(verified_drugs)
    if total_items == 0:
        summary_text = (
            "No recognized clinical test parameters or medication orders were found in the uploaded document."
        )
    else:
        abnormal_count = sum(1 for t in lab_tests if t.status in {"LOW", "HIGH"})
        critical_count = sum(1 for t in lab_tests if t.is_critical)

        summary_parts = [
            f"Analyzed {len(lab_tests)} laboratory measurement(s) and {len(verified_drugs)} medication order(s)."
        ]
        if critical_count > 0:
            summary_parts.append(
                f"⚠️ URGENT: {critical_count} critical value(s) require prompt physician evaluation."
            )
        elif abnormal_count > 0:
            summary_parts.append(
                f"Identified {abnormal_count} value(s) outside standard reference ranges."
            )
        else:
            summary_parts.append("All parsed measurements appear within standard reference intervals.")

        summary_text = " ".join(summary_parts)

    return ReportExplanationResult(
        document_summary=summary_text,
        explanations=explanations,
        unsupported_items=unsupported_items,
        disclaimer=STANDARD_REPORT_DISCLAIMER,
    )
