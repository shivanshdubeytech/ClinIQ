"""Medical Report and Prescription Analyzer Package.

Additive module for processing uploaded medical records, lab reports,
and prescriptions with verified retrieval and safety constraints.
"""

import os
from pathlib import Path
import site
import sys

# Auto-discover user and system site-packages if running inside minimal venv
_candidates = [
    site.getusersitepackages(),
    os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages"),
    os.path.expanduser(r"~\AppData\Roaming\Python\Python39\site-packages"),
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python39\Lib\site-packages"),
]
for _p in _candidates:
    if _p and _p not in sys.path and Path(_p).exists():
        sys.path.append(str(_p))

from .extractor import (
    ExtractedItem,
    ExtractionResult,
    assess_image_quality,
    extract_medical_document,
)
from .structurer import (
    LabTestItem,
    PrescriptionItem,
    StructuredDocument,
    parse_lab_line,
    parse_prescription_line,
    structure_medical_document,
)
from .comparator import (
    EvaluatedLabTest,
    STANDARD_REFERENCE_RANGES,
    compare_lab_report,
    evaluate_lab_item,
)
from .drug_verifier import (
    REAL_DRUG_DATABASE,
    VerifiedDrug,
    verify_drug_name,
    verify_prescriptions,
)
from .knowledge_base import (
    TRUSTED_MEDICAL_SOURCES,
    TrustedSource,
    retrieve_trusted_source,
)
from .explainer import (
    ItemExplanation,
    ReportExplanationResult,
    STANDARD_REPORT_DISCLAIMER,
    UNSUPPORTED_FALLBACK_TEXT,
    explain_analyzed_document,
    explain_lab_test,
    explain_prescription,
)
from .claim_verifier import (
    SecondPassVerificationResult,
    VerifiedClaim,
    apply_second_pass_verification,
    split_into_claims,
    verify_claim,
    verify_explanation_second_pass,
)
from .analyzer import (
    FinalAnalysisResult,
    LabeledReportItem,
    LABEL_CLEARLY_READ,
    LABEL_LOW_CONFIDENCE,
    LABEL_SEE_DOCTOR_SOON,
    analyze_medical_document,
    assign_item_label,
)

__all__ = [
    "ExtractedItem",
    "ExtractionResult",
    "assess_image_quality",
    "extract_medical_document",
    "LabTestItem",
    "PrescriptionItem",
    "StructuredDocument",
    "parse_lab_line",
    "parse_prescription_line",
    "structure_medical_document",
    "EvaluatedLabTest",
    "STANDARD_REFERENCE_RANGES",
    "compare_lab_report",
    "evaluate_lab_item",
    "REAL_DRUG_DATABASE",
    "VerifiedDrug",
    "verify_drug_name",
    "verify_prescriptions",
    "TRUSTED_MEDICAL_SOURCES",
    "TrustedSource",
    "retrieve_trusted_source",
    "ItemExplanation",
    "ReportExplanationResult",
    "STANDARD_REPORT_DISCLAIMER",
    "UNSUPPORTED_FALLBACK_TEXT",
    "explain_analyzed_document",
    "explain_lab_test",
    "explain_prescription",
    "SecondPassVerificationResult",
    "VerifiedClaim",
    "apply_second_pass_verification",
    "split_into_claims",
    "verify_claim",
    "verify_explanation_second_pass",
    "FinalAnalysisResult",
    "LabeledReportItem",
    "LABEL_CLEARLY_READ",
    "LABEL_LOW_CONFIDENCE",
    "LABEL_SEE_DOCTOR_SOON",
    "analyze_medical_document",
    "assign_item_label",
]
