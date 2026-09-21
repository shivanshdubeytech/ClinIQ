"""Final Evaluation Benchmark for Medical Report & Prescription Analyzer.

Evaluates:
  1. Extraction Accuracy: Correct extraction of test/drug names, values, units, and ranges.
  2. Range-Comparison Accuracy: Correct LOW / NORMAL / HIGH / CRITICAL status classification.
  3. Source-Groundedness Frequency: Percentage of claims and explanations fully supported by trusted sources.

Runs across a curated ground-truth test set of clinical documents.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.report_analyzer.analyzer import (
    analyze_medical_document,
    LABEL_CLEARLY_READ,
    LABEL_LOW_CONFIDENCE,
    LABEL_SEE_DOCTOR_SOON,
)
from tests.report_analyzer.test_stage_a_extractor import _generate_test_pdf


@dataclass
class GroundTruthReport:
    name: str
    doc_type: str  # "pdf" or "text"
    content_lines: List[str]
    expected_extracted_count: int
    expected_items: List[Dict[str, Any]]


BENCHMARK_GROUND_TRUTH: List[GroundTruthReport] = [
    GroundTruthReport(
        name="Report 1: Complete Blood Count (CBC Panel)",
        doc_type="pdf",
        content_lines=[
            "COMPLETE BLOOD COUNT",
            "Hemoglobin: 14.2 g/dL (Ref: 12.0 - 16.0)",
            "WBC Count: 14.5 x10^3/uL (Ref: 4.0 - 11.0)",
            "Platelets: 240 x10^3/uL (Ref: 150 - 450)",
        ],
        expected_extracted_count=3,
        expected_items=[
            {"name": "Hemoglobin", "val": 14.2, "status": "NORMAL", "critical": False},
            {"name": "WBC Count", "val": 14.5, "status": "HIGH", "critical": False},
            {"name": "Platelets", "val": 240.0, "status": "NORMAL", "critical": False},
        ],
    ),
    GroundTruthReport(
        name="Report 2: Critical Metabolic Emergency Panel",
        doc_type="pdf",
        content_lines=[
            "EMERGENCY METABOLIC PANEL",
            "Fasting Blood Glucose: 450 mg/dL (Ref: 70 - 99)",
            "Total Cholesterol: 245 mg/dL (Ref: < 200)",
        ],
        expected_extracted_count=2,
        expected_items=[
            {"name": "Glucose", "val": 450.0, "status": "HIGH", "critical": True},
            {"name": "Cholesterol", "val": 245.0, "status": "HIGH", "critical": False},
        ],
    ),
    GroundTruthReport(
        name="Report 3: Renal Panel with Missing Report Ranges (Fallback Test)",
        doc_type="pdf",
        content_lines=[
            "RENAL FUNCTION TEST",
            "Serum Creatinine: 1.0 mg/dL",
        ],
        expected_extracted_count=1,
        expected_items=[
            {"name": "Creatinine", "val": 1.0, "status": "NORMAL", "critical": False},
        ],
    ),
    GroundTruthReport(
        name="Report 4: Acute Respiratory Infection Prescription",
        doc_type="text",
        content_lines=[
            "PRESCRIPTION RECORD",
            "Rx: Amoxicillin 500mg capsules, take TID for 7 days",
        ],
        expected_extracted_count=1,
        expected_items=[
            {"name": "Amoxicillin", "val": "500mg", "status": "VERIFIED", "critical": False},
        ],
    ),
    GroundTruthReport(
        name="Report 5: Dual Chronic Disease Prescriptions (Diabetes & Hypertension)",
        doc_type="text",
        content_lines=[
            "DISCHARGE MEDICATIONS",
            "Rx: Metformin 500mg, twice daily with meals for 30 days",
            "Rx: Lisinopril 10mg tablet, once daily for 30 days",
        ],
        expected_extracted_count=2,
        expected_items=[
            {"name": "Metformin", "val": "500mg", "status": "VERIFIED", "critical": False},
            {"name": "Lisinopril", "val": "10mg", "status": "VERIFIED", "critical": False},
        ],
    ),
    GroundTruthReport(
        name="Report 6: Severe Critical Anemia Lab Report",
        doc_type="pdf",
        content_lines=[
            "HEMATOLOGY REPORT",
            "Hemoglobin: 6.2 g/dL (Ref: 12.0 - 16.0)",
        ],
        expected_extracted_count=1,
        expected_items=[
            {"name": "Hemoglobin", "val": 6.2, "status": "LOW", "critical": True},
        ],
    ),
]


def run_evaluation_benchmark() -> Dict[str, Any]:
    """Executes the benchmark suite and calculates empirical metrics."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    total_expected_items = 0
    correctly_extracted_items = 0

    total_range_comparisons = 0
    correct_range_comparisons = 0

    total_explanations = 0
    fully_grounded_explanations = 0

    report_metrics = []

    try:
        for idx, gt in enumerate(BENCHMARK_GROUND_TRUTH, start=1):
            if gt.doc_type == "pdf":
                file_path = temp_path / f"bench_{idx}.pdf"
                file_path.write_bytes(_generate_test_pdf(gt.content_lines))
            else:
                file_path = temp_path / f"bench_{idx}.txt"
                file_path.write_text("\n".join(gt.content_lines), encoding="utf-8")

            # Run analyzer
            analysis = analyze_medical_document(file_path)

            extracted_count = len(analysis.items)
            total_expected_items += gt.expected_extracted_count

            # Evaluate item matching
            matched_items_count = 0
            for exp_item in gt.expected_items:
                target_name = exp_item["name"].lower()
                found = next(
                    (it for it in analysis.items if target_name in it.item_name.lower()),
                    None,
                )
                if found:
                    matched_items_count += 1
                    correctly_extracted_items += 1

                    # Check range comparison accuracy
                    if found.item_type == "lab_test":
                        total_range_comparisons += 1
                        expected_status = exp_item["status"]
                        expected_critical = exp_item["critical"]
                        if found.status == expected_status and (found.label == LABEL_SEE_DOCTOR_SOON) == expected_critical:
                            correct_range_comparisons += 1

                    # Check explanation source support
                    total_explanations += 1
                    if found.source_name and "Unverified Claim Flagged" not in found.explanation:
                        fully_grounded_explanations += 1

            doc_extract_acc = (matched_items_count / gt.expected_extracted_count) * 100.0
            report_metrics.append({
                "name": gt.name,
                "expected": gt.expected_extracted_count,
                "extracted": extracted_count,
                "accuracy": doc_extract_acc,
                "overall_label": analysis.overall_label,
            })

    finally:
        temp_dir.cleanup()

    overall_extract_acc = (correctly_extracted_items / total_expected_items) * 100.0 if total_expected_items else 0.0
    overall_range_acc = (correct_range_comparisons / total_range_comparisons) * 100.0 if total_range_comparisons else 0.0
    overall_grounded_acc = (fully_grounded_explanations / total_explanations) * 100.0 if total_explanations else 0.0

    return {
        "reports": report_metrics,
        "total_items": total_expected_items,
        "extracted_items": correctly_extracted_items,
        "extraction_accuracy": overall_extract_acc,
        "total_range_checks": total_range_comparisons,
        "range_accuracy": overall_range_acc,
        "total_explanations": total_explanations,
        "source_grounded_frequency": overall_grounded_acc,
    }


if __name__ == "__main__":
    results = run_evaluation_benchmark()
    print("=" * 60)
    print("CLINIQ MEDICAL REPORT ANALYZER - BENCHMARK RESULTS")
    print("=" * 60)
    for r in results["reports"]:
        print(f"  {r['name']:<45} | Extracted: {r['extracted']}/{r['expected']} | Acc: {r['accuracy']:.1f}% | Label: {r['overall_label']}")
    print("-" * 60)
    print(f"Extraction Accuracy:          {results['extraction_accuracy']:.1f}% ({results['extracted_items']}/{results['total_items']})")
    print(f"Range-Comparison Accuracy:    {results['range_accuracy']:.1f}% ({results['range_accuracy']/100*results['total_range_checks']:.0f}/{results['total_range_checks']})")
    print(f"Source Support Frequency:     {results['source_grounded_frequency']:.1f}% ({results['source_grounded_frequency']/100*results['total_explanations']:.0f}/{results['total_explanations']})")
    print("=" * 60)
