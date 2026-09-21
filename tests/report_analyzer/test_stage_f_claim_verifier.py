"""Unit tests for Stage F: Second-Pass Claim Verifier against Trusted Sources.

Covers:
  Case 1: Fully supported factual claims pass with 1.0 groundedness
  Case 2: Unsupported / hallucinated claims are caught and flagged
  Case 3: Prohibited diagnostic statements stripped (Rule 7 enforcement)
  Case 4: Prohibited dosage alteration advice stripped (Rule 7 enforcement)
  Case 5: Messy / fragmented sentence parsing
  Case 6: Full document-level second pass verification workflow
"""

import unittest

from src.report_analyzer.claim_verifier import (
    apply_second_pass_verification,
    split_into_claims,
    verify_claim,
    verify_explanation_second_pass,
    SecondPassVerificationResult,
)
from src.report_analyzer.explainer import ItemExplanation, ReportExplanationResult
from src.report_analyzer.knowledge_base import retrieve_trusted_source


class TestStageFClaimVerifier(unittest.TestCase):
    """Test suite for factual claim verification and safety rule enforcement."""

    def test_01_fully_supported_claim(self):
        """Case 1: Fully factual claim directly supported by MedlinePlus source."""
        source = retrieve_trusted_source("Hemoglobin")
        self.assertIsNotNone(source)

        claim = "Hemoglobin is an iron-rich protein in red blood cells that carries oxygen from the lungs to tissues."
        v = verify_claim(claim, source.summary_content)

        self.assertTrue(v.is_supported)
        self.assertFalse(v.violates_safety_rule)
        self.assertGreater(v.confidence, 0.75)
        print("  [PASS] Case 1: Fully factual claim verified against MedlinePlus source.")

    def test_02_hallucinated_unsupported_claim(self):
        """Case 2: Fabricated or unsupported claim is flagged and not passed."""
        source = retrieve_trusted_source("Hemoglobin")
        self.assertIsNotNone(source)

        fake_claim = "Hemoglobin is synthesized by space asteroids and instantly cures baldness."
        v = verify_claim(fake_claim, source.summary_content)

        self.assertFalse(v.is_supported)
        self.assertFalse(v.violates_safety_rule)
        self.assertLess(v.confidence, 0.30)
        print("  [PASS] Case 2: Hallucinated claim successfully caught as unsupported.")

    def test_03_prohibited_diagnostic_statement(self):
        """Case 3: Hard Rule 7 violation - Prohibited diagnostic claim stripped."""
        source = retrieve_trusted_source("Fasting Blood Glucose")
        diag_claim = "This confirms you have severe type 2 diabetes and pancreatic failure."

        v = verify_claim(diag_claim, source.summary_content)
        self.assertFalse(v.is_supported)
        self.assertTrue(v.violates_safety_rule)
        self.assertIn("diagnostic statement", v.safety_reason.lower())
        print("  [PASS] Case 3: Rule 7 diagnostic statement detected and blocked.")

    def test_04_prohibited_dosage_modification(self):
        """Case 4: Hard Rule 7 violation - Prohibited dosage alteration advice stripped."""
        source = retrieve_trusted_source("Amoxicillin")
        dose_claim = "Increase your dose to 1000mg twice daily immediately."

        v = verify_claim(dose_claim, source.summary_content)
        self.assertFalse(v.is_supported)
        self.assertTrue(v.violates_safety_rule)
        self.assertIn("dosage modification", v.safety_reason.lower())
        print("  [PASS] Case 4: Rule 7 dosage change advice detected and blocked.")

    def test_05_explanation_second_pass_sanitization(self):
        """Case 5: Second pass on an explanation strips safety violations and flags unverified facts."""
        explanation = ItemExplanation(
            item_name="Hemoglobin",
            item_type="lab_test",
            measured_or_ordered="10.5 g/dL (LOW)",
            explanation_text=(
                "Hemoglobin carries oxygen in red blood cells throughout your body. "
                "You are diagnosed with chronic anemia and bone marrow exhaustion. "
                "Hemoglobin is made of liquid kryptonite that radiates energy."
            ),
            source_name="MedlinePlus",
            is_retrieved=True,
        )
        res = verify_explanation_second_pass(explanation)

        self.assertTrue(res.has_safety_violations)
        self.assertTrue(res.has_unsupported_claims)
        # Verify prohibited diagnosis was stripped
        self.assertNotIn("diagnosed with", res.verified_text.lower())
        # Verify hallucinated claim was flagged
        self.assertIn("Unverified Claim Flagged", res.verified_text)
        self.assertIn("carries oxygen", res.verified_text)
        print("  [PASS] Case 5: Explanation second pass cleanly sanitized prohibited & unverified text.")

    def test_06_document_level_verification(self):
        """Case 6: Full document-level second pass verification."""
        clean_exp = ItemExplanation(
            item_name="Metformin",
            item_type="prescription",
            measured_or_ordered="500mg daily",
            explanation_text="Metformin is an oral biguanide medication that improves insulin sensitivity.",
            source_name="MedlinePlus Drug Information",
            is_retrieved=True,
        )
        doc_res = ReportExplanationResult(
            document_summary="Summary of patient records.",
            explanations=[clean_exp],
        )
        verified_doc = apply_second_pass_verification(doc_res)

        self.assertEqual(len(verified_doc.explanations), 1)
        self.assertIn("Metformin", verified_doc.explanations[0].explanation_text)
        print("  [PASS] Case 6: Document-level second-pass verification applied seamlessly.")


if __name__ == "__main__":
    unittest.main()
