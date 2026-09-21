"""Unit tests for Stage G: Output Labels & End-to-End Pipeline.

Covers:
  Case 1: High-confidence clean lab report -> 'clearly read' label
  Case 2: Critical panic lab report -> 'see a doctor soon' label
  Case 3: Blurry / low-contrast degraded image -> 'low confidence - please verify' label
  Case 4: Prescription with unverified drug -> 'low confidence - please verify' label
  Case 5: Empty 0-byte file rejection
  Case 6: Unsupported file extension rejection
  Case 7: Mandatory disclaimer verification across all results
"""

from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from src.report_analyzer.analyzer import (
    analyze_medical_document,
    LABEL_CLEARLY_READ,
    LABEL_LOW_CONFIDENCE,
    LABEL_SEE_DOCTOR_SOON,
    STANDARD_REPORT_DISCLAIMER,
)
from tests.report_analyzer.test_stage_a_extractor import _generate_test_pdf


class TestStageGAnalyzer(unittest.TestCase):
    """Test suite for output labeling and end-to-end analyzer execution."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_clean_report_clearly_read(self):
        """Case 1: Standard clean digital lab report gets 'clearly read' label."""
        pdf_file = self.dir_path / "clean_blood_panel.pdf"
        lines = [
            "LABORATORY REPORT - METABOLIC",
            "Hemoglobin: 14.0 g/dL (Ref: 12.0 - 16.0)",
            "Total Cholesterol: 180 mg/dL (Ref: < 200)",
        ]
        pdf_file.write_bytes(_generate_test_pdf(lines))

        result = analyze_medical_document(pdf_file)

        self.assertTrue(result.is_success)
        self.assertEqual(result.overall_label, LABEL_CLEARLY_READ)
        self.assertFalse(result.has_critical_findings)
        self.assertIn("clearly read", result.items[0].label)
        self.assertIn(STANDARD_REPORT_DISCLAIMER, result.disclaimer)
        self.assertIn("### 📋 Lab Test Results", result.user_confirmation_table)
        print(f"  [PASS] Case 1: Clean report assigned label: '{result.overall_label}'.")

    def test_02_critical_findings_see_doctor_soon(self):
        """Case 2: Dangerous panic laboratory value gets 'see a doctor soon' label."""
        pdf_file = self.dir_path / "critical_labs.pdf"
        lines = [
            "URGENT LAB REPORT",
            "Fasting Blood Glucose: 450 mg/dL (Ref: 70 - 99)",
            "Hemoglobin: 13.0 g/dL (Ref: 12.0 - 16.0)",
        ]
        pdf_file.write_bytes(_generate_test_pdf(lines))

        result = analyze_medical_document(pdf_file)

        self.assertTrue(result.is_success)
        self.assertTrue(result.has_critical_findings)
        self.assertEqual(result.overall_label, LABEL_SEE_DOCTOR_SOON)
        # Verify the glucose item specifically has 'see a doctor soon'
        glucose_item = next(it for it in result.items if "glucose" in it.item_name.lower())
        self.assertEqual(glucose_item.label, LABEL_SEE_DOCTOR_SOON)
        print(f"  [PASS] Case 2: Critical finding assigned label: '{result.overall_label}'.")

    def test_03_blurry_image_low_confidence(self):
        """Case 3: Low-contrast blurry image triggers 'low confidence - please verify'."""
        img_file = self.dir_path / "blurry_report.png"
        img = Image.new("L", (120, 120), color=128)
        img.save(img_file)

        result = analyze_medical_document(img_file)

        # Image without OCR or blurry triggers low confidence
        self.assertFalse(result.is_success)
        self.assertEqual(result.overall_label, LABEL_LOW_CONFIDENCE)
        print(f"  [PASS] Case 3: Blurry/degraded image assigned label: '{result.overall_label}'.")

    def test_04_unverified_drug_low_confidence(self):
        """Case 4: Prescription containing unverified/fictional drug triggers 'low confidence'."""
        rx_file = self.dir_path / "fake_rx.txt"
        rx_file.write_text(
            "Rx: FakeUnapprovedPill 500mg, twice daily for 7 days",
            encoding="utf-8",
        )

        result = analyze_medical_document(rx_file)

        self.assertTrue(result.is_success)
        self.assertEqual(result.overall_label, LABEL_LOW_CONFIDENCE)
        self.assertEqual(result.items[0].label, LABEL_LOW_CONFIDENCE)
        print(f"  [PASS] Case 4: Unverified drug assigned label: '{result.overall_label}'.")

    def test_05_empty_file_rejected(self):
        """Case 5: Empty 0-byte file rejected with low confidence error."""
        empty_file = self.dir_path / "empty.pdf"
        empty_file.write_bytes(b"")

        result = analyze_medical_document(empty_file)

        self.assertFalse(result.is_success)
        self.assertEqual(result.overall_label, LABEL_LOW_CONFIDENCE)
        self.assertIn("empty", result.error.lower())
        print(f"  [PASS] Case 5: Empty file rejected gracefully with error: '{result.error}'.")

    def test_06_unsupported_format_rejected(self):
        """Case 6: Unsupported file extension rejected cleanly."""
        bad_file = self.dir_path / "payload.exe"
        bad_file.write_bytes(b"MZ\x00")

        result = analyze_medical_document(bad_file)

        self.assertFalse(result.is_success)
        self.assertIn("unsupported file format", result.error.lower())
        print(f"  [PASS] Case 6: Unsupported extension rejected with error: '{result.error}'.")

    def test_07_mandatory_disclaimer_present(self):
        """Case 7: Disclaimer is present on all final outputs."""
        txt_file = self.dir_path / "simple_rx.txt"
        txt_file.write_text("Rx: Amoxicillin 500mg capsules, take TID for 7 days", encoding="utf-8")

        result = analyze_medical_document(txt_file)
        self.assertTrue(result.is_success)
        self.assertIn("Educational Notice", result.disclaimer)
        self.assertIn("not a medical diagnosis", result.disclaimer)
        self.assertIn(STANDARD_REPORT_DISCLAIMER, result.summary_markdown)
        print("  [PASS] Case 7: Mandatory educational disclaimer verified.")


if __name__ == "__main__":
    unittest.main()
