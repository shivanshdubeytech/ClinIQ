"""Tests for clinical document structuring into tabular models."""
import pytest
from src.report_analyzer.extractor import ExtractedItem, ExtractionResult
from src.report_analyzer.structurer import structure_medical_document


def _make_extraction(lines):
    items = [ExtractedItem(text=line, confidence=0.95, source_type="text") for line in lines]
    return ExtractionResult(
        raw_text="\n".join(lines),
        items=items,
        overall_confidence=0.95,
        document_type="lab_report",
        page_count=1,
    )


def test_parse_hemoglobin_line():
    lines = ["Hemoglobin: 14.5 g/dL (Normal Range: 13.8 - 17.2)"]
    extraction = _make_extraction(lines)
    doc = structure_medical_document(extraction)
    assert len(doc.lab_tests) == 1
    assert doc.lab_tests[0].test_name.lower() == "hemoglobin"
    assert doc.lab_tests[0].value == 14.5
    assert doc.lab_tests[0].unit == "g/dL"


def test_parse_line_with_reference_range():
    lines = ["Potassium 4.2 mEq/L 3.5 - 5.0"]
    extraction = _make_extraction(lines)
    doc = structure_medical_document(extraction)
    assert len(doc.lab_tests) == 1
    test = doc.lab_tests[0]
    assert test.test_name.lower() == "potassium"
    assert test.value == 4.2
    assert test.ref_range_low == 3.5
    assert test.ref_range_high == 5.0


def test_parse_prescription_with_dose_and_frequency():
    lines = ["Metformin 500mg orally twice daily with meals for 30 days"]
    extraction = _make_extraction(lines)
    extraction.document_type = "prescription"
    doc = structure_medical_document(extraction)
    assert len(doc.prescriptions) == 1
    rx = doc.prescriptions[0]
    assert rx.drug_name.lower() == "metformin"
    assert "500" in rx.dose


def test_unsupported_line_is_skipped():
    lines = ["Hospital Registration Desk - Thank you for visiting"]
    extraction = _make_extraction(lines)
    doc = structure_medical_document(extraction)
    assert len(doc.lab_tests) == 0
    assert len(doc.prescriptions) == 0
