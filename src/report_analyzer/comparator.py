"""Stage C: Plain Python Reference Range Comparator & Critical Panic Value Detector.

Evaluates lab test values against reference ranges strictly using deterministic Python logic.
Language models are NEVER used to evaluate low/normal/high classifications or clinical ranges.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .structurer import LabTestItem, StructuredDocument


def _normalize_unit(unit: str) -> str:
    """Normalizes unit strings for robust comparison."""
    if not unit:
        return ""
    u = unit.strip().lower()
    u = re.sub(r"\s+", "", u)
    u = u.replace("micro", "u").replace("μ", "u")
    u = u.replace("10^3/ul", "x10^3/ul").replace("10*3/ul", "x10^3/ul").replace("k/ul", "x10^3/ul")
    u = u.replace("10^3/mm3", "x10^3/ul").replace("cells/mc", "x10^3/ul")
    return u


_UNIT_EQUIVALENCE = {
    "mg/dl": {"mg/dl", "mg/100ml", "mg%"},
    "g/dl": {"g/dl", "gm/dl", "gm%", "g%"},
    "x10^3/ul": {"x10^3/ul", "10^3/ul", "k/ul", "thou/ul", "/ul", "cells/ul", "cumm", "cells/cumm", "x10^3/mm3", "10^3/mm3"},
    "meq/l": {"meq/l", "mmol/l"},
    "%": {"%"},
    "uiu/ml": {"uiu/ml", "miu/l", "uunit/ml", "uunits/ml", "uu/ml", "miu/ml"},
}


def _units_compatible(unit_a: str, unit_b: str) -> bool:
    """Checks if two units are clinically compatible for numerical comparison."""
    if not unit_a or not unit_b:
        return False
    na = _normalize_unit(unit_a)
    nb = _normalize_unit(unit_b)
    if na == nb:
        return True
    for eq_set in _UNIT_EQUIVALENCE.values():
        if na in eq_set and nb in eq_set:
            return True
    return False


# Standard Clinical Reference Guidelines (Used ONLY when an uploaded report does not specify ranges)
# Based on established clinical reference manuals (Merck Manual, MedlinePlus, Harrison's)
STANDARD_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "hemoglobin": {
        "aliases": ["hemoglobin", "hgb", "hb"],
        "unit": "g/dL",
        "low": 12.0,
        "high": 17.5,
        "critical_low": 7.0,
        "critical_high": 20.0,
    },
    "wbc count": {
        "aliases": ["wbc", "wbc count", "white blood cell count", "leukocytes"],
        "unit": "x10^3/uL",
        "low": 4.0,
        "high": 11.0,
        "critical_low": 2.0,
        "critical_high": 30.0,
    },
    "platelets": {
        "aliases": ["platelets", "platelet count", "plt"],
        "unit": "x10^3/uL",
        "low": 150.0,
        "high": 450.0,
        "critical_low": 50.0,
        "critical_high": 1000.0,
    },
    "fasting blood glucose": {
        "aliases": ["glucose", "fasting blood glucose", "fbs", "blood sugar", "fasting glucose"],
        "unit": "mg/dL",
        "low": 70.0,
        "high": 99.0,
        "critical_low": 50.0,
        "critical_high": 400.0,
    },
    "total cholesterol": {
        "aliases": ["total cholesterol", "cholesterol"],
        "unit": "mg/dL",
        "low": None,
        "high": 200.0,
        "critical_low": None,
        "critical_high": 350.0,
    },
    "hdl cholesterol": {
        "aliases": ["hdl", "hdl cholesterol", "good cholesterol"],
        "unit": "mg/dL",
        "low": 40.0,
        "high": None,
        "critical_low": 20.0,
        "critical_high": None,
    },
    "ldl cholesterol": {
        "aliases": ["ldl", "ldl cholesterol", "bad cholesterol"],
        "unit": "mg/dL",
        "low": None,
        "high": 100.0,
        "critical_low": None,
        "critical_high": 220.0,
    },
    "serum creatinine": {
        "aliases": ["creatinine", "serum creatinine"],
        "unit": "mg/dL",
        "low": 0.6,
        "high": 1.3,
        "critical_low": None,
        "critical_high": 4.0,
    },
    "potassium": {
        "aliases": ["potassium", "k", "serum potassium"],
        "unit": "mEq/L",
        "low": 3.5,
        "high": 5.2,
        "critical_low": 2.8,
        "critical_high": 6.2,
    },
    "sodium": {
        "aliases": ["sodium", "na", "serum sodium"],
        "unit": "mEq/L",
        "low": 135.0,
        "high": 145.0,
        "critical_low": 120.0,
        "critical_high": 160.0,
    },
    "hba1c": {
        "aliases": ["hba1c", "glycated hemoglobin", "a1c"],
        "unit": "%",
        "low": 4.0,
        "high": 5.6,
        "critical_low": None,
        "critical_high": 11.0,
    },
    "tsh": {
        "aliases": ["tsh", "thyroid stimulating hormone"],
        "unit": "uIU/mL",
        "low": 0.4,
        "high": 4.5,
        "critical_low": 0.05,
        "critical_high": 15.0,
    },
}


@dataclass
class EvaluatedLabTest:
    """Represents a laboratory test evaluated strictly via Python logic."""
    test_name: str
    value: float
    unit: str
    ref_range_low: Optional[float]
    ref_range_high: Optional[float]
    ref_range_source: str  # "report", "standard_guidelines", "none"
    status: str  # "LOW", "NORMAL", "HIGH", "UNKNOWN_RANGE"
    is_critical: bool = False
    critical_alert: Optional[str] = None
    confidence: float = 1.0
    interpretation: str = ""
    raw_line: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "value": self.value,
            "unit": self.unit,
            "ref_range_low": self.ref_range_low,
            "ref_range_high": self.ref_range_high,
            "ref_range_source": self.ref_range_source,
            "status": self.status,
            "is_critical": self.is_critical,
            "critical_alert": self.critical_alert,
            "confidence": round(self.confidence, 3),
            "interpretation": self.interpretation,
            "raw_line": self.raw_line,
        }


def _match_standard_range(test_name: str) -> Optional[Dict[str, Any]]:
    """Matches a test name against known standard reference ranges."""
    normalized = test_name.strip().lower()
    for key, spec in STANDARD_REFERENCE_RANGES.items():
        for alias in spec["aliases"]:
            if alias in normalized or normalized in alias:
                return spec
    return None


def evaluate_lab_item(item: LabTestItem) -> EvaluatedLabTest:
    """Evaluates a single lab item using 100% deterministic Python comparisons.

    Do NOT use language models to perform numeric range comparison.
    """
    val = item.value
    low = item.ref_range_low
    high = item.ref_range_high
    source = "report" if (low is not None or high is not None) else "none"

    matched_spec = _match_standard_range(item.test_name)

    # Unit compatibility check against standard reference guidelines
    unit_compatible_with_standard = False
    if matched_spec and item.unit:
        unit_compatible_with_standard = _units_compatible(item.unit, matched_spec.get("unit", ""))

    # If report lacks range, fall back to standard clinical guidelines if matched and units are compatible
    unit_mismatch_fallback = False
    if low is None and high is None and matched_spec:
        if unit_compatible_with_standard:
            low = matched_spec["low"]
            high = matched_spec["high"]
            source = "standard_guidelines"
        else:
            unit_mismatch_fallback = True
            source = "unverified_unit"

    status = "UNKNOWN_RANGE"
    interpretation = ""

    if unit_mismatch_fallback:
        status = "UNVERIFIED_UNIT"
        expected_unit = matched_spec.get("unit", "")
        interpretation = (
            f"{item.test_name} value ({val} {item.unit}) could not be verified against standard reference range "
            f"({expected_unit}). Reference ranges vary by unit system. Consult your physician."
        )
    # Pure Python comparison logic
    elif low is not None and high is not None:
        if val < low:
            status = "LOW"
            interpretation = (
                f"{item.test_name} ({val} {item.unit}) is below the reference range "
                f"({low} - {high} {item.unit})."
            )
        elif val > high:
            status = "HIGH"
            interpretation = (
                f"{item.test_name} ({val} {item.unit}) is above the reference range "
                f"({low} - {high} {item.unit})."
            )
        else:
            status = "NORMAL"
            interpretation = (
                f"{item.test_name} ({val} {item.unit}) is within the normal reference range "
                f"({low} - {high} {item.unit})."
            )

    elif high is not None and low is None:
        # Upper bound only (e.g. Total Cholesterol < 200)
        if val > high:
            status = "HIGH"
            interpretation = (
                f"{item.test_name} ({val} {item.unit}) exceeds the reference upper limit of {high} {item.unit}."
            )
        else:
            status = "NORMAL"
            interpretation = (
                f"{item.test_name} ({val} {item.unit}) is within the reference limit (< {high} {item.unit})."
            )

    elif low is not None and high is None:
        # Lower bound only (e.g. HDL > 40)
        if val < low:
            status = "LOW"
            interpretation = (
                f"{item.test_name} ({val} {item.unit}) is below the recommended minimum of {low} {item.unit}."
            )
        else:
            status = "NORMAL"
            interpretation = (
                f"{item.test_name} ({val} {item.unit}) is within the reference threshold (> {low} {item.unit})."
            )
    else:
        status = "UNKNOWN_RANGE"
        interpretation = f"{item.test_name} is {val} {item.unit}. No reference range available."

    # Critical panic value detection (deterministic check)
    # Never apply standard panic thresholds if units are incompatible or unverified
    is_critical = False
    critical_alert = None

    if matched_spec and unit_compatible_with_standard and status != "UNVERIFIED_UNIT":
        crit_low = matched_spec.get("critical_low")
        crit_high = matched_spec.get("critical_high")
        if crit_low is not None and val <= crit_low:
            is_critical = True
            critical_alert = (
                f"CRITICAL VALUE: {item.test_name} ({val} {item.unit}) is dangerously low "
                f"(critical threshold <= {crit_low} {matched_spec.get('unit', '')}). Immediate medical evaluation recommended."
            )
        elif crit_high is not None and val >= crit_high:
            is_critical = True
            critical_alert = (
                f"CRITICAL VALUE: {item.test_name} ({val} {item.unit}) is dangerously high "
                f"(critical threshold >= {crit_high} {matched_spec.get('unit', '')}). Immediate medical evaluation recommended."
            )

    return EvaluatedLabTest(
        test_name=item.test_name,
        value=val,
        unit=item.unit,
        ref_range_low=low,
        ref_range_high=high,
        ref_range_source=source,
        status=status,
        is_critical=is_critical,
        critical_alert=critical_alert,
        confidence=item.confidence,
        interpretation=interpretation,
        raw_line=item.raw_line,
    )


def compare_lab_report(document: StructuredDocument) -> List[EvaluatedLabTest]:
    """Evaluates all lab tests in a StructuredDocument using deterministic Python logic.

    Args:
        document: StructuredDocument containing parsed LabTestItems.

    Returns:
        List of EvaluatedLabTest with low/normal/high status and critical alert flags.
    """
    return [evaluate_lab_item(test) for test in document.lab_tests]
