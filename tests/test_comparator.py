"""Tests for lab value comparator and reference range evaluation."""
import pytest
from src.report_analyzer.comparator import evaluate_lab_item
from src.report_analyzer.structurer import LabTestItem


def test_normal_value_is_normal():
    item = LabTestItem(test_name="Glucose", value=90.0, value_raw="90.0", unit="mg/dL")
    res = evaluate_lab_item(item)
    assert res.status == "NORMAL"
    assert not res.is_critical


def test_critical_low_fires_alert():
    item = LabTestItem(test_name="Glucose", value=35.0, value_raw="35.0", unit="mg/dL")
    res = evaluate_lab_item(item)
    assert res.is_critical
    assert res.status == "LOW"
    assert res.critical_alert is not None


def test_critical_high_fires_alert():
    item = LabTestItem(test_name="Potassium", value=6.5, value_raw="6.5", unit="mEq/L")
    res = evaluate_lab_item(item)
    assert res.is_critical
    assert res.status == "HIGH"
    assert res.critical_alert is not None


def test_mmol_l_glucose_is_unverified_not_critical():
    """Regression test: Glucose 5.5 mmol/L must not evaluate against US mg/dL ranges (70-99)."""
    item = LabTestItem(test_name="Glucose", value=5.5, value_raw="5.5", unit="mmol/L")
    res = evaluate_lab_item(item)
    assert res.status == "UNVERIFIED_UNIT"
    assert not res.is_critical


def test_report_supplied_range_is_trusted_as_is():
    item = LabTestItem(
        test_name="Custom Biomarker",
        value=15.0,
        value_raw="15.0",
        unit="U/L",
        ref_range_low=10.0,
        ref_range_high=20.0,
        ref_range_raw="10 - 20",
    )
    res = evaluate_lab_item(item)
    assert res.status == "NORMAL"
    assert res.ref_range_low == 10.0
    assert res.ref_range_high == 20.0


def test_unknown_unit_returns_unverified_status():
    item = LabTestItem(test_name="Hemoglobin", value=14.0, value_raw="14.0", unit="lightyears")
    res = evaluate_lab_item(item)
    assert res.status == "UNVERIFIED_UNIT"
    assert not res.is_critical
