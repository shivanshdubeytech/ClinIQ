"""Tests for prescription drug verification against authentic clinical datasets."""
import pytest
from src.report_analyzer.drug_verifier import verify_prescriptions
from src.report_analyzer.structurer import PrescriptionItem, StructuredDocument


def _make_rx_doc(drug_names):
    items = [
        PrescriptionItem(drug_name=name, dose="500mg", frequency="daily", duration="30 days")
        for name in drug_names
    ]
    return StructuredDocument(document_type="prescription", lab_tests=[], prescriptions=items)


def test_exact_generic_match():
    doc = _make_rx_doc(["Metformin"])
    verified = verify_prescriptions(doc)
    assert len(verified) == 1
    assert verified[0].is_verified
    assert verified[0].matched_generic.lower() == "metformin"


def test_exact_brand_match():
    doc = _make_rx_doc(["Glucophage", "Lipitor"])
    verified = verify_prescriptions(doc)
    assert len(verified) == 2
    assert verified[0].is_verified
    assert verified[0].matched_generic.lower() == "metformin"
    assert verified[1].is_verified
    assert verified[1].matched_generic.lower() == "atorvastatin"


def test_fuzzy_typo_match():
    doc = _make_rx_doc(["Atorvastatn"])  # common single-character typo
    verified = verify_prescriptions(doc)
    assert len(verified) == 1
    assert verified[0].is_verified
    assert verified[0].matched_generic.lower() == "atorvastatin"


def test_unrecognized_drug_returns_unverified():
    doc = _make_rx_doc(["FakeMedicationXYZ123"])
    verified = verify_prescriptions(doc)
    assert len(verified) == 1
    assert not verified[0].is_verified
    assert verified[0].matched_generic is None
