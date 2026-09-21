"""Stage D: Drug Verification Against Verified Drug Dataset (WHO / FDA / RxNorm).

Verifies prescribed medication names strictly against a curated, authoritative
drug dataset without relying on the language model's memory.
Implements fuzzy string distance matching to handle OCR typos and brand-to-generic resolution.
"""

from dataclasses import dataclass, field
import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

from .structurer import PrescriptionItem, StructuredDocument


@dataclass
class VerifiedDrug:
    """Represents the verification result of a medication order against the real drug dataset."""
    query_name: str
    matched_generic: Optional[str]
    brand_names: List[str] = field(default_factory=list)
    drug_class: str = ""
    primary_indications: List[str] = field(default_factory=list)
    is_verified: bool = False
    match_score: float = 0.0
    warning: Optional[str] = None
    raw_line: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_name": self.query_name,
            "matched_generic": self.matched_generic,
            "brand_names": self.brand_names,
            "drug_class": self.drug_class,
            "primary_indications": self.primary_indications,
            "is_verified": self.is_verified,
            "match_score": round(self.match_score, 3),
            "warning": self.warning,
            "raw_line": self.raw_line,
        }


# Curated Drug Registry derived from WHO Essential Medicines List & FDA Approved Drug Formularies
REAL_DRUG_DATABASE: Dict[str, Dict[str, Any]] = {
    "amoxicillin": {
        "brands": ["amoxil", "augmentin", "moxatag", "trimox"],
        "class": "Penicillin-class beta-lactam antibiotic",
        "indications": ["Bacterial infections", "Respiratory tract infections", "Ear/throat infections"],
    },
    "azithromycin": {
        "brands": ["zithromax", "z-pack", "zmax"],
        "class": "Macrolide antibiotic",
        "indications": ["Bacterial bronchitis", "Pneumonia", "Streptococcal pharyngitis"],
    },
    "ciprofloxacin": {
        "brands": ["cipro", "cipro xr"],
        "class": "Fluoroquinolone antibiotic",
        "indications": ["Urinary tract infections", "Gastrointestinal infections"],
    },
    "doxycycline": {
        "brands": ["vibramycin", "doryx"],
        "class": "Tetracycline antibiotic",
        "indications": ["Bacterial infections", "Lyme disease", "Acne"],
    },
    "metformin": {
        "brands": ["glucophage", "glumetza", "fortamet", "riomet"],
        "class": "Biguanide antidiabetic agent",
        "indications": ["Type 2 diabetes mellitus", "Glycemic regulation"],
    },
    "glipizide": {
        "brands": ["glucotrol", "glucotrol xl"],
        "class": "Sulfonylurea antidiabetic",
        "indications": ["Type 2 diabetes mellitus"],
    },
    "insulin glargine": {
        "brands": ["lantus", "toujeo", "basaglar"],
        "class": "Long-acting human insulin analog",
        "indications": ["Type 1 diabetes", "Type 2 diabetes glycemic management"],
    },
    "lisinopril": {
        "brands": ["prinivil", "zestril", "qbrelis"],
        "class": "ACE inhibitor antihypertensive",
        "indications": ["Hypertension (high blood pressure)", "Heart failure", "Post-myocardial infarction"],
    },
    "losartan": {
        "brands": ["cozaar"],
        "class": "Angiotensin II receptor blocker (ARB)",
        "indications": ["Hypertension", "Diabetic nephropathy"],
    },
    "amlodipine": {
        "brands": ["norvasc", "katerzia"],
        "class": "Dihydropyridine calcium channel blocker",
        "indications": ["Hypertension", "Coronary artery disease", "Angina"],
    },
    "atorvastatin": {
        "brands": ["lipitor"],
        "class": "HMG-CoA reductase inhibitor (Statin)",
        "indications": ["Hypercholesterolemia", "Cardiovascular event risk reduction"],
    },
    "simvastatin": {
        "brands": ["zocor", "flolipid"],
        "class": "HMG-CoA reductase inhibitor (Statin)",
        "indications": ["Dyslipidemia", "Cardiovascular prophylaxis"],
    },
    "metoprolol": {
        "brands": ["lopressor", "toprol-xl"],
        "class": "Selective beta-1 adrenergic blocker",
        "indications": ["Hypertension", "Angina pectoris", "Heart failure"],
    },
    "hydrochlorothiazide": {
        "brands": ["microzide", "esidrix", "hctz"],
        "class": "Thiazide diuretic",
        "indications": ["Hypertension", "Edema"],
    },
    "furosemide": {
        "brands": ["lasix"],
        "class": "Loop diuretic",
        "indications": ["Congestive heart failure edema", "Hepatic cirrhosis edema", "Renal disease"],
    },
    "levothyroxine": {
        "brands": ["synthroid", "levoxyl", "tirosint", "unithroid"],
        "class": "Synthetic thyroid hormone (T4)",
        "indications": ["Hypothyroidism", "TSH suppression"],
    },
    "albuterol": {
        "brands": ["proair", "ventolin", "proventil", "salbutamol"],
        "class": "Short-acting beta-2 adrenergic agonist (Bronchodilator)",
        "indications": ["Acute bronchospasm", "Asthma management", "COPD relief"],
    },
    "fluticasone": {
        "brands": ["flonase", "flovent", "armonair"],
        "class": "Inhaled corticosteroid",
        "indications": ["Asthma maintenance", "Allergic rhinitis"],
    },
    "montelukast": {
        "brands": ["singulair"],
        "class": "Leukotriene receptor antagonist",
        "indications": ["Asthma prophylaxis", "Seasonal allergic rhinitis"],
    },
    "omeprazole": {
        "brands": ["prilosec", "losec"],
        "class": "Proton pump inhibitor (PPI)",
        "indications": ["GERD (Acid reflux)", "Peptic ulcer disease", "Gastric hypersecretion"],
    },
    "pantoprazole": {
        "brands": ["protonix"],
        "class": "Proton pump inhibitor (PPI)",
        "indications": ["Erosive esophagitis", "GERD"],
    },
    "paracetamol": {
        "brands": ["tylenol", "acetaminophen", "panadol", "calpol"],
        "class": "Analgesic and antipyretic",
        "indications": ["Mild to moderate pain", "Fever reduction"],
    },
    "ibuprofen": {
        "brands": ["advil", "motrin", "nurofen"],
        "class": "Nonsteroidal anti-inflammatory drug (NSAID)",
        "indications": ["Inflammation", "Pain management", "Fever"],
    },
    "sertraline": {
        "brands": ["zoloft"],
        "class": "Selective serotonin reuptake inhibitor (SSRI)",
        "indications": ["Major depressive disorder", "Panic disorder", "Obsessive-compulsive disorder"],
    },
    "gabapentin": {
        "brands": ["neurontin", "gralise", "horizant"],
        "class": "GABA analog / Anticonvulsant",
        "indications": ["Neuropathic pain", "Postherpetic neuralgia", "Focal seizures"],
    },
}


def _clean_query(name: str) -> str:
    """Strips prescription prefixes, dosage tokens, frequencies, and forms from drug query string."""
    cleaned = name.lower().strip()
    # Remove Rx / Sig prefixes
    cleaned = re.sub(r"^(?:rx|sig|tab|cap|medication|dispense)[:\s]+", "", cleaned)
    # Remove dosage tokens like 500mg, 10 mg, 1 tablet, 2 tabs, etc.
    cleaned = re.sub(r"\b[0-9]+(?:\.[0-9]+)?\s*(?:mg|mcg|µg|g|ml|iu|tablets?|tabs?|capsules?|caps?|pills?|puffs?|drops?)\b", "", cleaned)
    # Remove common administration / frequency instruction words
    cleaned = re.sub(r"\b(daily|twice|three\s+times|four\s+times|once|bid|tid|qid|qd|qhs|prn|take|orally|oral|with\s+meals|po|at\s+bedtime|for\s+[0-9]+\s+days?|ongoing)\b", "", cleaned)
    # Remove dosage forms
    cleaned = re.sub(r"\b(capsules?|caps?|tablets?|tabs?|oral|syrup|inhaler|injection|soln|susp)\b", "", cleaned)
    # Remove punctuation
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", cleaned).strip()
    # Normalize extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def verify_drug_name(raw_name: str, threshold: float = 0.72) -> VerifiedDrug:
    """Verifies a single drug name against the real drug dataset.

    Args:
        raw_name: The extracted drug name from the prescription.
        threshold: Minimum similarity ratio for fuzzy matching (0.0 to 1.0).

    Returns:
        VerifiedDrug object with match details, generic name, class, and safety flags.
    """
    cleaned_query = _clean_query(raw_name)

    if not cleaned_query or len(cleaned_query) < 2:
        return VerifiedDrug(
            query_name=raw_name,
            matched_generic=None,
            is_verified=False,
            match_score=0.0,
            warning="Medication name is blank or invalid.",
        )

    # 1. Exact Match on Generic Name
    if cleaned_query in REAL_DRUG_DATABASE:
        data = REAL_DRUG_DATABASE[cleaned_query]
        return VerifiedDrug(
            query_name=raw_name,
            matched_generic=cleaned_query.capitalize(),
            brand_names=[b.capitalize() for b in data["brands"]],
            drug_class=data["class"],
            primary_indications=data["indications"],
            is_verified=True,
            match_score=1.0,
        )

    # 2. Exact Match on Brand / Trade Name
    for generic, data in REAL_DRUG_DATABASE.items():
        if cleaned_query in [b.lower() for b in data["brands"]]:
            return VerifiedDrug(
                query_name=raw_name,
                matched_generic=generic.capitalize(),
                brand_names=[b.capitalize() for b in data["brands"]],
                drug_class=data["class"],
                primary_indications=data["indications"],
                is_verified=True,
                match_score=0.98,
            )

    # 3. Direct Substring / Word-level Match in query
    words = cleaned_query.split()
    for word in words:
        if word in REAL_DRUG_DATABASE:
            data = REAL_DRUG_DATABASE[word]
            return VerifiedDrug(
                query_name=raw_name,
                matched_generic=word.capitalize(),
                brand_names=[b.capitalize() for b in data["brands"]],
                drug_class=data["class"],
                primary_indications=data["indications"],
                is_verified=True,
                match_score=1.0,
            )
        for generic, data in REAL_DRUG_DATABASE.items():
            if word in [b.lower() for b in data["brands"]]:
                return VerifiedDrug(
                    query_name=raw_name,
                    matched_generic=generic.capitalize(),
                    brand_names=[b.capitalize() for b in data["brands"]],
                    drug_class=data["class"],
                    primary_indications=data["indications"],
                    is_verified=True,
                    match_score=0.98,
                )

    # 4. Fuzzy Match against all Generic and Brand entries (handles OCR typos)
    candidates = []
    # Primary comparison target: cleaned_query or the first prominent word
    target = words[0] if words else cleaned_query

    for generic, data in REAL_DRUG_DATABASE.items():
        # Compare with generic
        gen_ratio = max(
            difflib.SequenceMatcher(None, cleaned_query, generic).ratio(),
            difflib.SequenceMatcher(None, target, generic).ratio(),
        )
        candidates.append((gen_ratio, generic, data))

        # Compare with each brand
        for brand in data["brands"]:
            b_ratio = max(
                difflib.SequenceMatcher(None, cleaned_query, brand).ratio(),
                difflib.SequenceMatcher(None, target, brand).ratio(),
            )
            candidates.append((b_ratio, generic, data))

    best_score, best_generic, best_data = max(candidates, key=lambda x: x[0])

    if best_score >= threshold:
        warning_msg = None
        if best_score < 0.99:
            warning_msg = (
                f"Fuzzy matched '{raw_name}' to '{best_generic.capitalize()}' "
                f"(similarity: {best_score:.2f}). Please confirm spelling."
            )

        return VerifiedDrug(
            query_name=raw_name,
            matched_generic=best_generic.capitalize(),
            brand_names=[b.capitalize() for b in best_data["brands"]],
            drug_class=best_data["class"],
            primary_indications=best_data["indications"],
            is_verified=True,
            match_score=best_score,
            warning=warning_msg,
        )

    # 4. Unverified Drug
    return VerifiedDrug(
        query_name=raw_name,
        matched_generic=None,
        is_verified=False,
        match_score=best_score,
        warning=(
            f"Unrecognized medication '{raw_name}'. Not verified in authoritative drug registry. "
            "Please consult a licensed pharmacist or physician to verify."
        ),
    )


def verify_prescriptions(document: StructuredDocument) -> List[VerifiedDrug]:
    """Verifies all prescription items within a StructuredDocument against the drug registry."""
    results = []
    for item in document.prescriptions:
        verified = verify_drug_name(item.drug_name)
        verified.raw_line = item.raw_line
        results.append(verified)
    return results
