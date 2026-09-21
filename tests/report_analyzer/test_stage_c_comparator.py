"""Unit tests for Stage C: Plain Python Reference Range Comparator.

Verifies deterministic Python comparison logic (NO language model involved):
  Case 1: Normal measurements within reference ranges
  Case 2: Low values below lower bound (LOW)
  Case 3: High values above upper bound (HIGH)
  Case 4: Critical panic values triggering immediate safety alerts
  Case 5: Fallback to standard clinical reference database when report omits ranges
  Case 6: Missing / unknown reference ranges handled gracefully
  Case 7: End-to-end test from messy/noisy extraction to structured comparison
"""

import unittest

from src.report_analyzer.comparator import (
    compare_lab_report,
    evaluate_lab_item,
    EvaluatedLabTest,
)
from src.report_analyzer.extractor import ExtractedItem, ExtractionResult
from src.report_analyzer.structurer import LabTestItem, structure_medical_document


class TestStageCComparator(unittest.TestCase):
    """Test suite for deterministic Python range comparison."""

    def test_01_normal_values_in_range(self):
        """Case 1: Normal values strictly within lower and upper bounds."""
        item1 = LabTestItem(
            test_name="Hemoglobin",
            value=14.2,
            value_raw="14.2",
            unit="g/dL",
            ref_range_low=12.0,
            ref_range_high=16.0,
            ref_range_raw="12.0 - 16.0",
        )
        res1 = evaluate_lab_item(item1)
        self.assertEqual(res1.status, "NORMAL")
        self.assertFalse(res1.is_critical)
        self.assertIn("within the normal reference range", res1.interpretation)

        # Upper bound only test
        item2 = LabTestItem(
            test_name="Total Cholesterol",
            value=175.0,
            value_raw="175",
            unit="mg/dL",
            ref_range_low=None,
            ref_range_high=200.0,
            ref_range_raw="< 200",
        )
        res2 = evaluate_lab_item(item2)
        self.assertEqual(res2.status, "NORMAL")
        self.assertFalse(res2.is_critical)

        print("  [PASS] Case 1: Normal values verified with deterministic Python logic.")

    def test_02_low_values(self):
        """Case 2: Values strictly below the lower reference bound."""
        item = LabTestItem(
            test_name="Hemoglobin",
            value=10.2,
            value_raw="10.2",
            unit="g/dL",
            ref_range_low=12.0,
            ref_range_high=16.0,
            ref_range_raw="12.0 - 16.0",
        )
        res = evaluate_lab_item(item)
        self.assertEqual(res.status, "LOW")
        self.assertIn("below the reference range", res.interpretation)
        print("  [PASS] Case 2: Low values evaluated accurately as LOW.")

    def test_03_high_values(self):
        """Case 3: Values strictly exceeding the upper reference bound."""
        item = LabTestItem(
            test_name="Fasting Blood Glucose",
            value=135.0,
            value_raw="135",
            unit="mg/dL",
            ref_range_low=70.0,
            ref_range_high=99.0,
            ref_range_raw="70 - 99",
        )
        res = evaluate_lab_item(item)
        self.assertEqual(res.status, "HIGH")
        self.assertIn("above the reference range", res.interpretation)
        print("  [PASS] Case 3: High values evaluated accurately as HIGH.")

    def test_04_critical_panic_values(self):
        """Case 4: Dangerous clinical thresholds triggering critical alerts."""
        # Extremely high glucose
        crit_glucose = LabTestItem(
            test_name="Fasting Blood Glucose",
            value=450.0,
            value_raw="450",
            unit="mg/dL",
            ref_range_low=70.0,
            ref_range_high=99.0,
        )
        res_g = evaluate_lab_item(crit_glucose)
        self.assertEqual(res_g.status, "HIGH")
        self.assertTrue(res_g.is_critical)
        self.assertIsNotNone(res_g.critical_alert)
        self.assertIn("dangerously high", res_g.critical_alert)

        # Dangerously low hemoglobin
        crit_hgb = LabTestItem(
            test_name="Hemoglobin",
            value=6.2,
            value_raw="6.2",
            unit="g/dL",
            ref_range_low=12.0,
            ref_range_high=16.0,
        )
        res_h = evaluate_lab_item(crit_hgb)
        self.assertEqual(res_h.status, "LOW")
        self.assertTrue(res_h.is_critical)
        self.assertIn("dangerously low", res_h.critical_alert)

        print("  [PASS] Case 4: Critical panic values triggered safety flags.")

    def test_05_fallback_to_standard_guidelines(self):
        """Case 5: When an uploaded report omits ranges, fallback to standard guidelines."""
        item_no_range = LabTestItem(
            test_name="Serum Creatinine",
            value=1.1,
            value_raw="1.1",
            unit="mg/dL",
            ref_range_low=None,
            ref_range_high=None,
            ref_range_raw="",
        )
        res = evaluate_lab_item(item_no_range)
        self.assertEqual(res.ref_range_source, "standard_guidelines")
        self.assertEqual(res.status, "NORMAL")
        self.assertEqual(res.ref_range_low, 0.6)
        self.assertEqual(res.ref_range_high, 1.3)
        print("  [PASS] Case 5: Fallback to standard clinical reference guidelines verified.")

    def test_06_unknown_test_no_range(self):
        """Case 6: Unrecognized proprietary biomarker with no available range."""
        item_unknown = LabTestItem(
            test_name="Proprietary Inflammatory Index X99",
            value=42.0,
            value_raw="42",
            unit="units",
            ref_range_low=None,
            ref_range_high=None,
        )
        res = evaluate_lab_item(item_unknown)
        self.assertEqual(res.status, "UNKNOWN_RANGE")
        self.assertFalse(res.is_critical)
        self.assertIn("No reference range available", res.interpretation)
        print("  [PASS] Case 6: Unknown test with no reference range handled safely.")

    def test_07_end_to_end_structure_and_compare(self):
        """Case 7: End-to-end evaluation of full lab report."""
        report_text = (
            "Hemoglobin: 13.5 g/dL (Ref: 12.0 - 16.0)\n"
            "WBC Count: 14.5 x10^3/uL (Ref: 4.0 - 11.0)\n"
            "Platelets: 80 x10^3/uL (Ref: 150 - 450)\n"
        )
        doc = structure_medical_document(report_text)
        evaluated = compare_lab_report(doc)

        self.assertEqual(len(evaluated), 3)
        self.assertEqual(evaluated[0].status, "NORMAL")
        self.assertEqual(evaluated[1].status, "HIGH")
        self.assertEqual(evaluated[2].status, "LOW")
        print("  [PASS] Case 7: End-to-end structured lab evaluation verified.")


if __name__ == "__main__":
    unittest.main()
