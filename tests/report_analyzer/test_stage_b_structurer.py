"""Unit tests for Stage B: Document Structuring into Tables (Lab Reports & Prescriptions).

Covers:
  Case 1: Standard clinical lab panel with units and reference ranges
  Case 2: Multi-drug prescription order (name, dose, frequency, duration)
  Case 3: Mixed clinical summary containing both lab tests and prescriptions
  Case 4: Messy / noisy / OCR-degraded input with ambiguous formatting
  Case 5: Empty or completely non-medical text input
  Case 6: User confirmation Markdown table rendering
"""

import unittest

from src.report_analyzer.extractor import ExtractedItem, ExtractionResult
from src.report_analyzer.structurer import (
    LabTestItem,
    PrescriptionItem,
    StructuredDocument,
    parse_lab_line,
    parse_prescription_line,
    structure_medical_document,
)


class TestStageBStructurer(unittest.TestCase):
    """Test suite for parsing unstructured medical text into structured tables."""

    def test_01_standard_lab_panel(self):
        """Case 1: Standard clinical lab panel with values, units, and ranges."""
        report_text = (
            "PATIENT LAB RESULTS - CBC & METABOLIC\n"
            "Hemoglobin: 13.5 g/dL (Ref: 12.0 - 16.0)\n"
            "WBC Count: 7.2 x10^3/uL (Ref: 4.0 - 11.0)\n"
            "Platelets: 240 x10^3/uL (Ref: 150 - 450)\n"
            "Fasting Blood Glucose: 110 mg/dL (Ref: 70 - 99)\n"
            "Total Cholesterol: 215 mg/dL (Ref: < 200)\n"
        )
        doc = structure_medical_document(report_text)

        self.assertEqual(doc.document_type, "lab_report")
        self.assertEqual(len(doc.lab_tests), 5)
        self.assertEqual(len(doc.prescriptions), 0)

        # Verify Hemoglobin details
        hgb = doc.lab_tests[0]
        self.assertIn("Hemoglobin", hgb.test_name)
        self.assertEqual(hgb.value, 13.5)
        self.assertEqual(hgb.unit, "g/dL")
        self.assertEqual(hgb.ref_range_low, 12.0)
        self.assertEqual(hgb.ref_range_high, 16.0)

        # Verify Cholesterol upper-bound reference range
        chol = doc.lab_tests[4]
        self.assertIn("Cholesterol", chol.test_name)
        self.assertEqual(chol.value, 215.0)
        self.assertEqual(chol.ref_range_high, 200.0)
        self.assertIsNone(chol.ref_range_low)

        print("  [PASS] Case 1: Standard lab panel parsed into structured LabTestItems.")

    def test_02_multi_drug_prescription(self):
        """Case 2: Multi-medication prescription order."""
        rx_text = (
            "PRESCRIPTION ORDER\n"
            "Rx: Amoxicillin 500mg capsules, take 1 capsule orally three times daily for 7 days\n"
            "Rx: Metformin 500 mg, twice daily with meals for 1 month\n"
            "Rx: Lisinopril 10mg tablet, once daily in the morning ongoing\n"
        )
        doc = structure_medical_document(rx_text)

        self.assertEqual(doc.document_type, "prescription")
        self.assertEqual(len(doc.prescriptions), 3)
        self.assertEqual(len(doc.lab_tests), 0)

        # Check Amoxicillin
        amox = doc.prescriptions[0]
        self.assertEqual(amox.drug_name, "Amoxicillin")
        self.assertEqual(amox.dose, "500mg")
        self.assertIn("three times daily", amox.frequency)
        self.assertIn("7 days", amox.duration)

        # Check Lisinopril
        lis = doc.prescriptions[2]
        self.assertEqual(lis.drug_name, "Lisinopril")
        self.assertEqual(lis.dose, "10mg")
        self.assertIn("once daily", lis.frequency)

        print("  [PASS] Case 2: Prescription orders parsed into structured PrescriptionItems.")

    def test_03_mixed_clinical_summary(self):
        """Case 3: Mixed clinical note with both lab test results and prescriptions."""
        mixed_text = (
            "HOSPITAL DISCHARGE NOTE\n"
            "Serum Creatinine: 1.1 mg/dL (Ref: 0.6 - 1.2)\n"
            "HbA1c: 6.8 % (Ref: 4.0 - 5.6)\n"
            "Discharge Med: Atorvastatin 20mg once daily at bedtime for 30 days\n"
        )
        doc = structure_medical_document(mixed_text)

        self.assertEqual(doc.document_type, "mixed")
        self.assertEqual(len(doc.lab_tests), 2)
        self.assertEqual(len(doc.prescriptions), 1)

        creat = doc.lab_tests[0]
        self.assertIn("Creatinine", creat.test_name)
        self.assertEqual(creat.value, 1.1)

        med = doc.prescriptions[0]
        self.assertIn("Atorvastatin", med.drug_name)
        self.assertEqual(med.dose, "20mg")

        print("  [PASS] Case 3: Mixed clinical summary correctly identified and segregated.")

    def test_04_messy_noisy_ocr_input(self):
        """Case 4: Messy / noisy text with corrupted tokens and low extraction confidence."""
        noisy_items = [
            ExtractedItem(text="Lab Rprt ***", confidence=0.4, source_type="image_ocr"),
            ExtractedItem(text="Glucose  95  mg/dL", confidence=0.6, source_type="image_ocr"),
            ExtractedItem(text="??# random blotch ::", confidence=0.2, source_type="image_ocr"),
            ExtractedItem(text="Metformin 500mg daily", confidence=0.55, source_type="image_ocr"),
        ]
        noisy_extraction = ExtractionResult(
            raw_text="\n".join(it.text for it in noisy_items),
            items=noisy_items,
            overall_confidence=0.45,
            document_type="image",
            is_blurry_or_low_quality=True,
        )

        doc = structure_medical_document(noisy_extraction)

        self.assertGreaterEqual(len(doc.lab_tests) + len(doc.prescriptions), 1)
        self.assertGreaterEqual(len(doc.unparsed_lines), 1)
        # Verify confidence was propagated from low-quality OCR
        for test in doc.lab_tests:
            self.assertLess(test.confidence, 0.75)
        for rx in doc.prescriptions:
            self.assertLess(rx.confidence, 0.75)

        print(f"  [PASS] Case 4: Noisy OCR input handled with degraded confidence scores and unparsed lines.")

    def test_05_empty_or_non_medical_input(self):
        """Case 5: Empty string and non-medical greeting."""
        empty_doc = structure_medical_document("")
        self.assertEqual(empty_doc.document_type, "unknown")
        self.assertEqual(len(empty_doc.lab_tests), 0)
        self.assertEqual(len(empty_doc.prescriptions), 0)

        non_med_doc = structure_medical_document("Good morning! This is just a personal letter with no health info.")
        self.assertEqual(non_med_doc.document_type, "unknown")
        self.assertEqual(len(non_med_doc.lab_tests), 0)

        print("  [PASS] Case 5: Empty and non-medical input classified safely as unknown.")

    def test_06_markdown_table_rendering(self):
        """Case 6: User confirmation Markdown table rendering."""
        report_text = (
            "Hemoglobin: 13.5 g/dL (Ref: 12.0 - 16.0)\n"
            "Rx: Amoxicillin 500mg, three times daily for 7 days\n"
        )
        doc = structure_medical_document(report_text)
        md_table = doc.to_markdown_table()

        self.assertIn("### 📋 Lab Test Results", md_table)
        self.assertIn("Hemoglobin", md_table)
        self.assertIn("13.5", md_table)
        self.assertIn("### 💊 Prescribed Medications", md_table)
        self.assertIn("Amoxicillin", md_table)
        self.assertIn("500mg", md_table)

        print("  [PASS] Case 6: User confirmation Markdown table formatted properly.")


if __name__ == "__main__":
    unittest.main()
