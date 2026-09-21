"""Unit tests for Stage A: Extractor and Confidence Scoring.

Covers:
  Case 1: Clean digital medical lab report PDF
  Case 2: Digital prescription note with medication instructions
  Case 3: Messy / blurry / low-contrast image report
  Case 4: Empty 0-byte file
  Case 5: Wrong / unsupported file extension
  Case 6: Privacy auto-deletion of temporary uploaded files (Rule 8)
"""

import io
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from src.report_analyzer.extractor import (
    assess_image_quality,
    extract_medical_document,
    ExtractionResult,
)


def _generate_test_pdf(text_lines) -> bytes:
    """Generates a valid lightweight single-page PDF containing given text lines."""
    content_cmds = ["BT", "/F1 12 Tf"]
    y = 720
    for line in text_lines:
        safe_line = line.replace("(", "\\(").replace(")", "\\)")
        content_cmds.append(f"50 {y} Td ({safe_line}) Tj -50 -20 Td")
        y -= 20
    content_cmds.append("ET")
    stream = "\n".join(content_cmds)
    stream_len = len(stream.encode("latin-1"))

    pdf = (
        f"%PDF-1.4\n"
        f"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        f"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        f"4 0 obj << /Length {stream_len} >> stream\n"
        f"{stream}\n"
        f"endstream endobj\n"
        f"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        f"xref\n"
        f"0 6\n"
        f"0000000000 65535 f \n"
        f"0000000009 00000 n \n"
        f"0000000058 00000 n \n"
        f"0000000115 00000 n \n"
        f"0000000244 00000 n \n"
        f"0000000300 00000 n \n"
        f"trailer << /Size 6 /Root 1 0 R >>\n"
        f"startxref\n"
        f"370\n"
        f"%%EOF\n"
    )
    return pdf.encode("latin-1")


class TestStageAExtractor(unittest.TestCase):
    """Test suite for document extraction, OCR confidence scoring, and safety constraints."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_digital_pdf_lab_report(self):
        """Case 1: Clean digital medical report PDF should extract text with high confidence."""
        pdf_file = self.dir_path / "complete_blood_count.pdf"
        lines = [
            "CLINICAL LABORATORY REPORT",
            "Hemoglobin: 13.5 g/dL (Ref: 12.0 - 16.0)",
            "WBC Count: 7.2 x10^3/uL (Ref: 4.0 - 11.0)",
            "Platelets: 240 x10^3/uL (Ref: 150 - 450)",
            "Fasting Blood Glucose: 95 mg/dL (Ref: 70 - 99)",
        ]
        pdf_file.write_bytes(_generate_test_pdf(lines))

        result = extract_medical_document(pdf_file)

        self.assertTrue(result.is_success, f"Expected success, got error: {result.error}")
        self.assertEqual(result.document_type, "pdf")
        self.assertGreaterEqual(result.page_count, 1)
        self.assertGreater(len(result.items), 0)
        self.assertGreaterEqual(result.overall_confidence, 0.85)
        self.assertIn("Hemoglobin", result.raw_text)
        self.assertIn("Platelets", result.raw_text)
        print("  [PASS] Case 1: Digital PDF lab report extracted with high confidence.")

    def test_02_prescription_text_extract(self):
        """Case 2: Prescription document with dosage instructions."""
        rx_file = self.dir_path / "prescription_amoxicillin.txt"
        rx_content = (
            "Rx: Amoxicillin 500mg capsules\n"
            "Sig: Take 1 capsule orally three times daily for 7 days\n"
            "Dispense: 21 capsules\n"
            "Refills: 0\n"
            "Prescriber: Dr. S. Rao, MD"
        )
        rx_file.write_text(rx_content, encoding="utf-8")

        result = extract_medical_document(rx_file)

        self.assertTrue(result.is_success)
        self.assertEqual(result.document_type, "text")
        self.assertGreaterEqual(len(result.items), 4)
        self.assertIn("Amoxicillin 500mg", result.raw_text)
        self.assertGreaterEqual(result.overall_confidence, 0.90)
        print("  [PASS] Case 2: Prescription document extracted with valid confidence.")

    def test_03_messy_blurry_low_quality_image(self):
        """Case 3: Degraded, low-contrast, low-resolution blurry image."""
        img_file = self.dir_path / "blurry_rx.png"
        # Create a tiny 150x150 uniform low-contrast image (washed out gray)
        img = Image.new("L", (150, 150), color=128)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "faint text", fill=130)  # barely distinguishable contrast
        img.save(img_file)

        # Assess visual metrics directly
        is_low_quality, score, notes = assess_image_quality(img_file)
        self.assertTrue(is_low_quality)
        self.assertLess(score, 0.70)
        self.assertGreater(len(notes), 0)

        # Run full extraction entrypoint
        result = extract_medical_document(img_file)
        self.assertTrue(result.is_blurry_or_low_quality)
        self.assertLessEqual(result.overall_confidence, 0.50)
        print(f"  [PASS] Case 3: Blurry/low-quality image flagged properly (notes: {result.quality_notes}).")

    def test_04_empty_zero_byte_file(self):
        """Case 4: Empty 0-byte file handling."""
        empty_file = self.dir_path / "empty_report.pdf"
        empty_file.write_bytes(b"")

        result = extract_medical_document(empty_file)

        self.assertFalse(result.is_success)
        self.assertEqual(result.document_type, "empty")
        self.assertIn("empty", result.error.lower())
        print(f"  [PASS] Case 4: Empty file properly rejected with: '{result.error}'.")

    def test_05_unsupported_file_type(self):
        """Case 5: Unsupported / wrong file extension."""
        bad_file = self.dir_path / "malware.exe"
        bad_file.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00")

        result = extract_medical_document(bad_file)

        self.assertFalse(result.is_success)
        self.assertEqual(result.document_type, "invalid")
        self.assertIn("unsupported file format", result.error.lower())
        print(f"  [PASS] Case 5: Unsupported file extension rejected with: '{result.error}'.")

    def test_06_temporary_file_auto_deletion(self):
        """Case 6: Rule 8 privacy constraint - auto-delete temp file after processing."""
        temp_upload = self.dir_path / "temp_patient_record.pdf"
        temp_upload.write_bytes(_generate_test_pdf(["Patient Name: Anonymous", "Test: Lipid Panel"]))
        self.assertTrue(temp_upload.exists())

        # Extract with auto_delete_temp=True
        result = extract_medical_document(temp_upload, auto_delete_temp=True)

        self.assertTrue(result.is_success)
        # Ensure file was deleted after processing
        self.assertFalse(temp_upload.exists(), "Temporary medical upload was NOT deleted!")
        print("  [PASS] Case 6: Rule 8 verified - uploaded file safely deleted after extraction.")


if __name__ == "__main__":
    unittest.main()
