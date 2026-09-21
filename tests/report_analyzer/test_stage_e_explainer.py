"""Unit tests for Stage E: Plain-Language Medical Explanations with Strict Grounding.

Covers:
  Case 1: Grounded lab test explanation with MedlinePlus citation & doctor questions
  Case 2: Grounded prescription explanation with FDA/MedlinePlus citation
  Case 3: Unretrieved unknown item strictly returning 'I don't have reliable information on this.'
  Case 4: Critical panic value alert integration
  Case 5: Full document narrative Markdown rendering with disclaimer
  Case 6: Empty document with no parsed items handled safely
"""

import unittest

from src.report_analyzer.comparator import EvaluatedLabTest
from src.report_analyzer.drug_verifier import VerifiedDrug
from src.report_analyzer.explainer import (
    explain_analyzed_document,
    explain_lab_test,
    explain_prescription,
    ItemExplanation,
    ReportExplanationResult,
    STANDARD_REPORT_DISCLAIMER,
    UNSUPPORTED_FALLBACK_TEXT,
)
from src.report_analyzer.structurer import PrescriptionItem


class TestStageEExplainer(unittest.TestCase):
    """Test suite for grounded patient explanations over trusted sources."""

    def test_01_grounded_lab_explanation(self):
        """Case 1: Grounded lab test explanation with authoritative citation."""
        test_item = EvaluatedLabTest(
            test_name="Hemoglobin",
            value=10.5,
            unit="g/dL",
            ref_range_low=12.0,
            ref_range_high=16.0,
            ref_range_source="report",
            status="LOW",
            is_critical=False,
            interpretation="Hemoglobin (10.5 g/dL) is below the reference range (12.0 - 16.0 g/dL).",
        )
        res = explain_lab_test(test_item)

        self.assertTrue(res.is_retrieved)
        self.assertIn("MedlinePlus", res.source_name)
        self.assertIn("below the reference range", res.explanation_text)
        self.assertGreaterEqual(len(res.suggested_doctor_questions), 2)
        # Verify strict safety: No prescribing or dose adjustments
        self.assertNotIn("take mg", res.explanation_text.lower())
        self.assertNotIn("diagnosed with", res.explanation_text.lower())
        print("  [PASS] Case 1: Grounded lab test explanation cites MedlinePlus and includes doctor questions.")

    def test_02_grounded_prescription_explanation(self):
        """Case 2: Grounded medication explanation with trusted drug label source."""
        drug = VerifiedDrug(
            query_name="Amoxicillin",
            matched_generic="Amoxicillin",
            brand_names=["Amoxil", "Augmentin"],
            drug_class="Penicillin-class beta-lactam antibiotic",
            primary_indications=["Bacterial infections", "Respiratory tract infections"],
            is_verified=True,
            match_score=1.0,
        )
        rx_item = PrescriptionItem(
            drug_name="Amoxicillin",
            dose="500mg",
            frequency="three times daily",
            duration="7 days",
        )
        res = explain_prescription(drug, rx_item=rx_item)

        self.assertTrue(res.is_retrieved)
        self.assertIn("FDA", res.source_name)
        self.assertIn("Amoxicillin", res.explanation_text)
        self.assertIn("bacterial infections", res.explanation_text.lower())
        self.assertGreaterEqual(len(res.suggested_doctor_questions), 1)
        print("  [PASS] Case 2: Prescription explanation grounded in FDA Prescribing Information.")

    def test_03_unretrieved_item_strict_fallback(self):
        """Case 3: Unretrieved unknown item outputs strictly 'I don't have reliable information on this.'"""
        unknown_test = EvaluatedLabTest(
            test_name="NonExistentBiomarkerXYZ99",
            value=45.0,
            unit="units",
            ref_range_low=None,
            ref_range_high=None,
            ref_range_source="none",
            status="UNKNOWN_RANGE",
        )
        res = explain_lab_test(unknown_test)

        self.assertFalse(res.is_retrieved)
        self.assertEqual(res.explanation_text, UNSUPPORTED_FALLBACK_TEXT)
        self.assertIsNone(res.source_name)
        print(f"  [PASS] Case 3: Unretrieved item returned exact fallback: '{UNSUPPORTED_FALLBACK_TEXT}'.")

    def test_04_critical_panic_alert_present(self):
        """Case 4: Critical panic value alert is captured and emphasized."""
        crit_test = EvaluatedLabTest(
            test_name="Fasting Blood Glucose",
            value=450.0,
            unit="mg/dL",
            ref_range_low=70.0,
            ref_range_high=99.0,
            ref_range_source="report",
            status="HIGH",
            is_critical=True,
            critical_alert="CRITICAL VALUE: Fasting Blood Glucose (450.0 mg/dL) is dangerously high.",
            interpretation="Fasting Blood Glucose is above reference range.",
        )
        res = explain_lab_test(crit_test)

        self.assertTrue(res.is_retrieved)
        self.assertIsNotNone(res.critical_alert)
        self.assertIn("dangerously high", res.critical_alert)
        print("  [PASS] Case 4: Critical panic alert properly embedded in explanation.")

    def test_05_document_narrative_markdown(self):
        """Case 5: Full document narrative with findings, citations, and disclaimer."""
        lab_test = EvaluatedLabTest(
            test_name="Hemoglobin",
            value=13.5,
            unit="g/dL",
            ref_range_low=12.0,
            ref_range_high=16.0,
            ref_range_source="report",
            status="NORMAL",
            interpretation="Hemoglobin is normal.",
        )
        drug = VerifiedDrug(
            query_name="Metformin",
            matched_generic="Metformin",
            brand_names=["Glucophage"],
            drug_class="Biguanide antidiabetic",
            primary_indications=["Type 2 diabetes mellitus"],
            is_verified=True,
            match_score=1.0,
        )
        res = explain_analyzed_document(
            lab_tests=[lab_test],
            verified_drugs=[drug],
        )

        md = res.to_markdown()
        self.assertIn("## 📋 Plain-Language Summary", md)
        self.assertIn("Hemoglobin", md)
        self.assertIn("Metformin", md)
        self.assertIn("Source:", md)
        self.assertIn(STANDARD_REPORT_DISCLAIMER, md)
        print("  [PASS] Case 5: Full document Markdown narrative formatted cleanly with disclaimer.")

    def test_06_empty_document_safe_handling(self):
        """Case 6: Empty document with zero findings handled safely."""
        res = explain_analyzed_document()
        self.assertIn("No recognized clinical test parameters", res.document_summary)
        self.assertEqual(len(res.explanations), 0)
        self.assertEqual(len(res.unsupported_items), 0)
        print("  [PASS] Case 6: Empty document gracefully returns safe placeholder narrative.")


if __name__ == "__main__":
    unittest.main()
