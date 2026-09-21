"""Stage B: Document Structuring into Tables (Lab Reports & Prescriptions).

Parses raw extracted text from medical files into clean, validated tabular data:
  1. Lab Reports: (test_name, value, unit, ref_range_low, ref_range_high, ref_range_raw)
  2. Prescriptions: (drug_name, dose, frequency, duration)
Generates user confirmation tables in markdown and JSON format.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .extractor import ExtractedItem, ExtractionResult


@dataclass
class LabTestItem:
    """Represents a single parsed clinical laboratory test measurement."""
    test_name: str
    value: float
    value_raw: str
    unit: str
    ref_range_low: Optional[float] = None
    ref_range_high: Optional[float] = None
    ref_range_raw: str = ""
    confidence: float = 1.0
    raw_line: str = ""
    is_ambiguous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "value": self.value,
            "value_raw": self.value_raw,
            "unit": self.unit,
            "ref_range_low": self.ref_range_low,
            "ref_range_high": self.ref_range_high,
            "ref_range_raw": self.ref_range_raw,
            "confidence": round(self.confidence, 3),
            "is_ambiguous": self.is_ambiguous,
            "raw_line": self.raw_line,
        }


@dataclass
class PrescriptionItem:
    """Represents a single parsed medication prescription order."""
    drug_name: str
    dose: str
    frequency: str
    duration: str
    confidence: float = 1.0
    raw_line: str = ""
    is_ambiguous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_name": self.drug_name,
            "dose": self.dose,
            "frequency": self.frequency,
            "duration": self.duration,
            "confidence": round(self.confidence, 3),
            "is_ambiguous": self.is_ambiguous,
            "raw_line": self.raw_line,
        }


@dataclass
class StructuredDocument:
    """Encapsulates the structured output containing lab tests or prescriptions."""
    document_type: str  # "lab_report", "prescription", "mixed", "unknown"
    lab_tests: List[LabTestItem] = field(default_factory=list)
    prescriptions: List[PrescriptionItem] = field(default_factory=list)
    unparsed_lines: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    requires_user_confirmation: bool = True

    def to_markdown_table(self) -> str:
        """Renders an accessible Markdown table for user confirmation."""
        md_parts = []

        if self.lab_tests:
            md_parts.append("### 📋 Lab Test Results")
            md_parts.append("| Test Name | Value | Unit | Reference Range | Confidence | Note |")
            md_parts.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in self.lab_tests:
                conf_label = "✅ High" if t.confidence >= 0.85 else "⚠️ Verify"
                note = "Ambiguous" if t.is_ambiguous else "Normal Read"
                ref_display = t.ref_range_raw if t.ref_range_raw else "N/A"
                md_parts.append(
                    f"| **{t.test_name}** | {t.value_raw} | {t.unit or '-'} | {ref_display} | {conf_label} ({t.confidence:.2f}) | {note} |"
                )
            md_parts.append("")

        if self.prescriptions:
            md_parts.append("### 💊 Prescribed Medications")
            md_parts.append("| Medication | Dosage | Frequency | Duration | Confidence | Note |")
            md_parts.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for p in self.prescriptions:
                conf_label = "✅ High" if p.confidence >= 0.85 else "⚠️ Verify"
                note = "Ambiguous" if p.is_ambiguous else "Normal Read"
                md_parts.append(
                    f"| **{p.drug_name}** | {p.dose or 'N/A'} | {p.frequency or 'As directed'} | {p.duration or 'Not specified'} | {conf_label} ({p.confidence:.2f}) | {note} |"
                )
            md_parts.append("")

        if not self.lab_tests and not self.prescriptions:
            md_parts.append("*No recognized lab test values or prescriptions could be extracted.*")

        return "\n".join(md_parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "lab_tests": [t.to_dict() for t in self.lab_tests],
            "prescriptions": [p.to_dict() for p in self.prescriptions],
            "unparsed_lines_count": len(self.unparsed_lines),
            "overall_confidence": round(self.overall_confidence, 3),
            "requires_user_confirmation": self.requires_user_confirmation,
        }


# ==========================================
# Regular Expression Matchers & Extraction
# ==========================================

# Reference range patterns e.g. "12.0 - 16.0", "70-99", "< 200", "> 60", "4.0 to 11.0"
RANGE_PATTERN = re.compile(
    r"(?:ref(?:erence)?\s*(?:range)?[:\s]*)?"
    r"(?:\(|\[)?\s*"
    r"(?:(<|>|<=|>=)\s*([0-9]+(?:\.[0-9]+)?)|([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to)\s*([0-9]+(?:\.[0-9]+)?))"
    r"\s*(?:\)|\])?",
    re.IGNORECASE,
)

# Common lab measurement units
UNITS = [
    r"x10\^3/[uµ]L",
    r"x10\^6/[uµ]L",
    r"10\^3/[uµ]L",
    r"10\^6/[uµ]L",
    r"g/dL",
    r"mg/dL",
    r"µg/dL",
    r"ug/dL",
    r"mmol/L",
    r"umol/L",
    r"µmol/L",
    r"mEq/L",
    r"uIU/mL",
    r"µIU/mL",
    r"mIU/L",
    r"IU/L",
    r"U/L",
    r"ng/mL",
    r"pg/mL",
    r"cells/[uµ]L",
    r"%",
    r"fl",
    r"fL",
    r"pg",
    r"mm/hr",
    r"sec",
    r"seconds",
]
UNITS_REGEX = re.compile(r"\b(" + "|".join(UNITS) + r")\b", re.IGNORECASE)

# Common dosage units e.g. 500mg, 10 mg, 2.5 ml, 50mcg, 2 puffs, 1 tablet
DOSE_REGEX = re.compile(
    r"\b([0-9]+(?:\.[0-9]+)?)\s*(mg|mcg|µg|g|ml|mL|iu|IU|units|puffs?|drops?|tablets?|capsules?)\b",
    re.IGNORECASE,
)

# Frequency patterns
FREQ_REGEX = re.compile(
    r"\b("
    r"once\s+daily|twice\s+daily|three\s+times\s+daily|four\s+times\s+daily|"
    r"once\s+a\s+day|twice\s+a\s+day|three\s+times\s+a\s+day|"
    r"every\s+[0-9]+\s+hours?|every\s+morning|every\s+night|at\s+bedtime|"
    r"daily|BID|TID|QID|QD|QHS|PRN|as\s+needed"
    r")\b",
    re.IGNORECASE,
)

# Duration patterns
DURATION_REGEX = re.compile(
    r"\b("
    r"(?:for\s+)?[0-9]+\s+(?:days?|weeks?|months?)|"
    r"x\s*[0-9]+\s+(?:days?|weeks?|months?)|"
    r"until\s+finished|ongoing|long[- ]term"
    r")\b",
    re.IGNORECASE,
)


def _parse_reference_range(text_segment: str) -> Tuple[Optional[float], Optional[float], str]:
    """Extracts lower and upper bounds from reference range substrings."""
    match = RANGE_PATTERN.search(text_segment)
    if not match:
        return None, None, ""

    raw_range = match.group(0).strip(" ()[]")
    operator, single_val, low_val, high_val = match.groups()

    if operator and single_val:
        val = float(single_val)
        if "<" in operator:
            return None, val, raw_range
        elif ">" in operator:
            return val, None, raw_range

    if low_val and high_val:
        return float(low_val), float(high_val), raw_range

    return None, None, raw_range


def parse_lab_line(line: str, base_confidence: float = 1.0) -> Optional[LabTestItem]:
    """Attempts to parse a single line of text into a LabTestItem."""
    cleaned = line.strip()
    if len(cleaned) < 4:
        return None

    # Do not treat prescription lines as lab tests
    if re.search(r"^(?:Rx|Sig|Tab|Cap|Medication|Discharge Med)[:\s]", cleaned, re.IGNORECASE):
        return None

    # Look for reference range first (often at end of line)
    ref_low, ref_high, ref_raw = _parse_reference_range(cleaned)

    # Remove the reference range substring to avoid numeric confusion
    working_line = cleaned
    if ref_raw:
        working_line = cleaned.replace(ref_raw, " ").strip()

    # Look for measurement unit
    unit_match = UNITS_REGEX.search(working_line)
    unit = unit_match.group(0) if unit_match else ""

    # If line has frequency instructions (e.g. daily, twice daily) and no lab unit, it's medication
    if FREQ_REGEX.search(working_line) and not unit:
        return None

    # Split on colon or separators if present
    # Examples: "Hemoglobin: 13.5 g/dL" or "Platelets 240 x10^3/uL"
    test_name = ""
    measured_val = None
    value_raw = ""

    # Check for name : value format
    if ":" in working_line:
        parts = working_line.split(":", 1)
        potential_name = parts[0].strip(" -*#")
        rest = parts[1].strip()

        # Find first number in rest
        num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", rest)
        if num_match and len(potential_name) >= 2:
            test_name = potential_name
            value_raw = num_match.group(1)
            try:
                measured_val = float(value_raw)
            except ValueError:
                measured_val = None
    else:
        # Pattern: [Test Name Words] [Number] [Optional Unit]
        match = re.search(r"^([A-Za-z0-9\s\/\-\(\)]+?)\s+([0-9]+(?:\.[0-9]+)?)\s*(.*)$", working_line)
        if match:
            potential_name = match.group(1).strip(" -*#")
            potential_num = match.group(2)
            # Ensure name isn't just numbers or garbage
            if any(c.isalpha() for c in potential_name) and len(potential_name) >= 2:
                test_name = potential_name
                value_raw = potential_num
                try:
                    measured_val = float(value_raw)
                except ValueError:
                    measured_val = None

    if test_name and measured_val is not None:
        # Clean up test name
        test_name = re.sub(r"^(test|result|exam)\s*:\s*", "", test_name, flags=re.IGNORECASE).strip()
        if test_name.lower() in {"rx", "sig", "tab", "cap", "medication", "discharge med", "dose"}:
            return None
        is_ambiguous = (unit == "" and ref_raw == "") or base_confidence < 0.75
        item_conf = base_confidence if not is_ambiguous else max(0.4, base_confidence * 0.8)

        return LabTestItem(
            test_name=test_name,
            value=measured_val,
            value_raw=value_raw,
            unit=unit,
            ref_range_low=ref_low,
            ref_range_high=ref_high,
            ref_range_raw=ref_raw,
            confidence=item_conf,
            raw_line=cleaned,
            is_ambiguous=is_ambiguous,
        )

    return None


def parse_prescription_line(line: str, base_confidence: float = 1.0) -> Optional[PrescriptionItem]:
    """Attempts to parse a single line of text into a PrescriptionItem."""
    cleaned = line.strip()
    if len(cleaned) < 5:
        return None

    # Detect prescription prefixes: Rx:, Sig:, Tab, Cap, Medication, Discharge Med, or direct drug line
    is_rx_explicit = bool(re.search(r"^(?:Rx|Sig|Tab|Cap|Medication|Discharge Med|Dispense)[:\s]", cleaned, re.IGNORECASE))

    # Extract dose
    dose_match = DOSE_REGEX.search(cleaned)
    dose = dose_match.group(0) if dose_match else ""

    # Extract frequency
    freq_match = FREQ_REGEX.search(cleaned)
    freq = freq_match.group(0) if freq_match else ""

    # Extract duration
    dur_match = DURATION_REGEX.search(cleaned)
    dur = dur_match.group(0) if dur_match else ""

    # If it has a dose and a frequency or explicit Rx marker, extract drug name
    if (dose and freq) or is_rx_explicit or (dose and len(cleaned.split()) <= 6):
        working = re.sub(r"^(?:Rx|Sig|Tab|Cap|Medication|Discharge Med)[:\s]+", "", cleaned, flags=re.IGNORECASE).strip()

        # Drug name is usually the first 1-3 words before dosage
        if dose:
            name_part = working.split(dose)[0].strip(" ,-:")
        else:
            name_part = working.split(",")[0].strip()

        # Clean words
        drug_name = re.sub(r"\b(capsules?|tablets?|oral|po)\b", "", name_part, flags=re.IGNORECASE).strip(" ,-:")

        if drug_name and any(c.isalpha() for c in drug_name):
            is_ambiguous = not dose or not freq or base_confidence < 0.75
            item_conf = base_confidence if not is_ambiguous else max(0.4, base_confidence * 0.8)

            return PrescriptionItem(
                drug_name=drug_name,
                dose=dose,
                frequency=freq,
                duration=dur,
                confidence=item_conf,
                raw_line=cleaned,
                is_ambiguous=is_ambiguous,
            )

    return None


def structure_medical_document(
    input_data: Union[str, ExtractionResult],
) -> StructuredDocument:
    """Converts unstructured medical text into structured tables for user confirmation.

    Args:
        input_data: Extracted text string or ExtractionResult from Stage A.

    Returns:
        StructuredDocument containing classified lab tests and/or prescriptions.
    """
    if isinstance(input_data, ExtractionResult):
        items_source = input_data.items
        raw_text = input_data.raw_text
        doc_conf = input_data.overall_confidence
    else:
        raw_text = str(input_data)
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        items_source = [
            ExtractedItem(text=ln, confidence=1.0, source_type="text")
            for ln in lines
        ]
        doc_conf = 1.0 if items_source else 0.0

    lab_tests: List[LabTestItem] = []
    prescriptions: List[PrescriptionItem] = []
    unparsed_lines: List[str] = []

    for item in items_source:
        text = item.text.strip()
        conf = item.confidence

        # Try lab test parse
        lab_res = parse_lab_line(text, base_confidence=conf)
        if lab_res:
            lab_tests.append(lab_res)
            continue

        # Try prescription parse
        rx_res = parse_prescription_line(text, base_confidence=conf)
        if rx_res:
            prescriptions.append(rx_res)
            continue

        if len(text) > 3:
            unparsed_lines.append(text)

    # Determine classification
    if lab_tests and prescriptions:
        doc_type = "mixed"
    elif lab_tests:
        doc_type = "lab_report"
    elif prescriptions:
        doc_type = "prescription"
    else:
        doc_type = "unknown"

    # Compute overall confidence
    all_confidences = [t.confidence for t in lab_tests] + [p.confidence for p in prescriptions]
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else doc_conf

    return StructuredDocument(
        document_type=doc_type,
        lab_tests=lab_tests,
        prescriptions=prescriptions,
        unparsed_lines=unparsed_lines,
        overall_confidence=avg_conf,
        requires_user_confirmation=True,
    )
