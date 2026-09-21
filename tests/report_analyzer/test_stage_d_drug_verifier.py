"""Unit tests for Stage D: Drug Verification Against Authoritative Drug Dataset.

Verifies matching against real drug registries (NO model hallucination):
  Case 1: Exact generic name match (Amoxicillin, Metformin)
  Case 2: Brand name to generic resolution (Glucophage -> Metformin, Lipitor -> Atorvastatin)
  Case 3: Fuzzy matching with OCR typo correction ("Amoxicilin", "Lisinoprl")
  Case 4: Unrecognized / fictional drug rejection ("SuperCure-999")
  Case 5: Messy input with dosage tokens and punctuation
  Case 6: Blank or empty drug name input
  Case 7: Full StructuredDocument prescription verification
"""

import unittest

from src.report_analyzer.drug_verifier import (
    VerifiedDrug,
    verify_drug_name,
    verify_prescriptions,
)
from src.report_analyzer.structurer import PrescriptionItem, StructuredDocument


class TestStageDDrugVerifier(unittest.TestCase):
    """Test suite for drug dataset verification and typo tolerance."""

    def test_01_exact_generic_match(self):
        """Case 1: Exact generic match in drug registry."""
        res = verify_drug_name("Amoxicillin")
        self.assertTrue(res.is_verified)
        self.assertEqual(res.matched_generic, "Amoxicillin")
        self.assertEqual(res.match_score, 1.0)
        self.assertIn("antibiotic", res.drug_class.lower())
        self.assertGreater(len(res.primary_indications), 0)
        print("  [PASS] Case 1: Exact generic match verified against real dataset.")

    def test_02_brand_to_generic_resolution(self):
        """Case 2: Trade brand name resolves to proper generic entity."""
        res_glucophage = verify_drug_name("Glucophage")
        self.assertTrue(res_glucophage.is_verified)
        self.assertEqual(res_glucophage.matched_generic, "Metformin")

        res_lipitor = verify_drug_name("Lipitor")
        self.assertTrue(res_lipitor.is_verified)
        self.assertEqual(res_lipitor.matched_generic, "Atorvastatin")
        print("  [PASS] Case 2: Brand names (Glucophage, Lipitor) successfully resolved to generic counterparts.")

    def test_03_fuzzy_ocr_typo_correction(self):
        """Case 3: Typo / OCR degradation successfully resolved with fuzzy distance."""
        # Single letter omission: "Amoxicilin"
        res_typo = verify_drug_name("Amoxicilin")
        self.assertTrue(res_typo.is_verified)
        self.assertEqual(res_typo.matched_generic, "Amoxicillin")
        self.assertGreater(res_typo.match_score, 0.85)
        self.assertIsNotNone(res_typo.warning)

        # Truncated spelling: "Lisinoprl"
        res_lis = verify_drug_name("Lisinoprl")
        self.assertTrue(res_lis.is_verified)
        self.assertEqual(res_lis.matched_generic, "Lisinopril")

        print("  [PASS] Case 3: Fuzzy matching corrected OCR typos while logging safety confirmation warnings.")

    def test_04_unrecognized_fake_drug(self):
        """Case 4: Fictional or unapproved chemical rejected."""
        fake_name = "SuperMiracleMed-X100"
        res = verify_drug_name(fake_name)
        self.assertFalse(res.is_verified)
        self.assertIsNone(res.matched_generic)
        self.assertIn("not verified in authoritative drug registry", res.warning.lower())
        print("  [PASS] Case 4: Unrecognized drug safely rejected without inventing medical data.")

    def test_05_messy_input_with_dosages(self):
        """Case 5: Messy string containing dosages, forms, and punctuation."""
        raw = "Rx: Metformin 500mg capsules (2 tabs daily)"
        res = verify_drug_name(raw)
        self.assertTrue(res.is_verified)
        self.assertEqual(res.matched_generic, "Metformin")
        print("  [PASS] Case 5: Messy input with dosage tokens cleaned and matched.")

    def test_06_empty_or_blank_input(self):
        """Case 6: Empty string or whitespace input."""
        res_empty = verify_drug_name("")
        self.assertFalse(res_empty.is_verified)
        self.assertIsNone(res_empty.matched_generic)

        res_spaces = verify_drug_name("   ")
        self.assertFalse(res_spaces.is_verified)
        print("  [PASS] Case 6: Empty medication names rejected safely.")

    def test_07_structured_document_batch_verification(self):
        """Case 7: Full batch verification from StructuredDocument."""
        doc = StructuredDocument(
            document_type="prescription",
            prescriptions=[
                PrescriptionItem(drug_name="Amoxicillin", dose="500mg", frequency="TID", duration="7 days"),
                PrescriptionItem(drug_name="Lisinopril", dose="10mg", frequency="daily", duration="30 days"),
                PrescriptionItem(drug_name="UnknownPill99", dose="5mg", frequency="daily", duration="10 days"),
            ],
        )
        verified_list = verify_prescriptions(doc)
        self.assertEqual(len(verified_list), 3)
        self.assertTrue(verified_list[0].is_verified)
        self.assertTrue(verified_list[1].is_verified)
        self.assertFalse(verified_list[2].is_verified)
        print("  [PASS] Case 7: Batch document prescriptions verified successfully.")


if __name__ == "__main__":
    unittest.main()
