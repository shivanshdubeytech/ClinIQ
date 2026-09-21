"""Stage E: Curated Trusted Medical Knowledge Base (MedlinePlus, WHO, NIH, FDA).

Contains verified patient education excerpts and trusted retrieval matching.
Does NOT modify or touch existing ChromaDB collections (Hard Rule 5).
"""

from dataclasses import dataclass
import difflib
import re
from typing import Any, Dict, List, Optional


@dataclass
class TrustedSource:
    """Represents an authoritative retrieved source for explaining tests and medications."""
    topic: str
    source_name: str  # e.g., "MedlinePlus (U.S. National Library of Medicine)"
    source_url: str
    summary_content: str
    key_points: List[str]
    suggested_doctor_questions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "summary_content": self.summary_content,
            "key_points": self.key_points,
            "suggested_doctor_questions": self.suggested_doctor_questions,
        }


# Authoritative Medical Repository from NIH / MedlinePlus / WHO / FDA
TRUSTED_MEDICAL_SOURCES: Dict[str, TrustedSource] = {
    "hemoglobin": TrustedSource(
        topic="Hemoglobin",
        source_name="MedlinePlus (U.S. National Library of Medicine)",
        source_url="https://medlineplus.gov/lab-tests/hemoglobin-test/",
        summary_content=(
            "Hemoglobin is an iron-rich protein in red blood cells that carries oxygen from the lungs "
            "to tissues throughout the body. A low hemoglobin level usually indicates anemia, which "
            "can cause fatigue, weakness, or pale skin. High hemoglobin can occur with dehydration, "
            "smoking, lung conditions, or bone marrow disorders."
        ),
        key_points=[
            "Transports oxygen throughout body tissues",
            "Low values often indicate nutritional deficiency, blood loss, or anemia",
            "High values may correlate with dehydration or respiratory conditions",
        ],
        suggested_doctor_questions=[
            "Could my hemoglobin level explain my current energy levels or symptoms?",
            "Do I need any follow-up tests, such as iron studies or ferritin?",
            "Are there dietary considerations or lifestyle changes I should discuss?",
        ],
    ),
    "wbc count": TrustedSource(
        topic="White Blood Cell (WBC) Count",
        source_name="NIH National Cancer Institute / MedlinePlus",
        source_url="https://medlineplus.gov/lab-tests/white-blood-cell-count/",
        summary_content=(
            "White blood cells are part of the immune system that defend the body against infections and disease. "
            "An elevated WBC count (leukocytosis) frequently indicates an immune response to bacterial or viral infection, "
            "inflammation, physical stress, or certain medications. A low WBC count (leukopenia) means reduced immune defenses, "
            "which can be caused by viral infections, bone marrow issues, or autoimmune disorders."
        ),
        key_points=[
            "Measures immune system activity and defenses",
            "Elevated levels often reflect infection, inflammation, or physical stress",
            "Low levels suggest reduced capacity to fight off infections",
        ],
        suggested_doctor_questions=[
            "Does my WBC count indicate an active infection or inflammation?",
            "Should we repeat this test in a few weeks to monitor the trend?",
            "Are there precautions I should take regarding exposure to illness?",
        ],
    ),
    "platelets": TrustedSource(
        topic="Platelet Count",
        source_name="MedlinePlus (U.S. National Library of Medicine)",
        source_url="https://medlineplus.gov/lab-tests/platelet-tests/",
        summary_content=(
            "Platelets (thrombocytes) are tiny cell fragments essential for normal blood clotting and wound healing. "
            "A low platelet count (thrombocytopenia) increases the risk of bruising or bleeding. "
            "A high platelet count (thrombocytosis) can arise from reactive inflammation, iron deficiency, or marrow disorders."
        ),
        key_points=[
            "Essential for blood clotting and stopping bleeding",
            "Low levels increase susceptibility to bruising or prolonged bleeding",
            "High levels may be reactive to temporary inflammation or underlying conditions",
        ],
        suggested_doctor_questions=[
            "Do my platelet levels put me at any increased risk of bruising or bleeding?",
            "Could any of my current medications or supplements affect platelet counts?",
        ],
    ),
    "fasting blood glucose": TrustedSource(
        topic="Fasting Blood Glucose",
        source_name="NIH National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)",
        source_url="https://www.niddk.nih.gov/health-information/diagnostic-tests/blood-glucose-test",
        summary_content=(
            "Blood glucose measures the concentration of sugar in the bloodstream after an overnight fast. "
            "It serves as a primary screening tool for prediabetes and diabetes. Normal fasting levels are generally "
            "between 70 and 99 mg/dL. Values between 100 and 125 mg/dL indicate prediabetes, while fasting levels of 126 mg/dL "
            "or higher on multiple tests suggest diabetes."
        ),
        key_points=[
            "Reflects baseline blood sugar metabolism after fasting",
            "Values 100-125 mg/dL suggest prediabetes; 126+ mg/dL suggests diabetes",
            "Lifestyle, diet, and physical activity significantly influence fasting glucose",
        ],
        suggested_doctor_questions=[
            "Should I have an HbA1c test to assess my average blood sugar over the past 3 months?",
            "What dietary or physical activity habits would best support healthy glucose levels?",
        ],
    ),
    "total cholesterol": TrustedSource(
        topic="Total Cholesterol & Lipid Profile",
        source_name="National Heart, Lung, and Blood Institute (NHLBI / NIH)",
        source_url="https://www.nhlbi.nih.gov/health/blood-cholesterol",
        summary_content=(
            "Cholesterol is a waxy substance used to build cell membranes and produce hormones. "
            "Excess levels in the blood can accumulate along arterial walls, forming plaques that elevate "
            "the risk of cardiovascular disease. Desirable total cholesterol is generally below 200 mg/dL."
        ),
        key_points=[
            "Key biomarker for cardiovascular risk assessment",
            "Levels over 200 mg/dL may require lifestyle modification or lipid therapy",
            "Best interpreted alongside HDL, LDL, and triglycerides",
        ],
        suggested_doctor_questions=[
            "How does my total cholesterol relate to my overall 10-year cardiovascular risk?",
            "What dietary patterns (such as reducing saturated fats) do you advise for me?",
        ],
    ),
    "serum creatinine": TrustedSource(
        topic="Serum Creatinine & Kidney Function",
        source_name="National Kidney Foundation / MedlinePlus",
        source_url="https://medlineplus.gov/lab-tests/creatinine-test/",
        summary_content=(
            "Creatinine is a normal waste product produced by muscle breakdown and filtered out of the blood by the kidneys. "
            "Elevated creatinine levels indicate that the kidneys may not be filtering waste efficiently. "
            "Doctors use creatinine alongside age and sex to calculate the estimated Glomerular Filtration Rate (eGFR)."
        ),
        key_points=[
            "Primary biomarker of renal filtration efficiency",
            "Higher levels can signify kidney stress, dehydration, or acute injury",
            "Used to calculate eGFR to determine overall kidney health",
        ],
        suggested_doctor_questions=[
            "What is my estimated glomerular filtration rate (eGFR) based on this creatinine?",
            "Could temporary dehydration or exercise have influenced this reading?",
        ],
    ),
    "amoxicillin": TrustedSource(
        topic="Amoxicillin",
        source_name="FDA Prescribing Information / MedlinePlus Drug Information",
        source_url="https://medlineplus.gov/druginfo/meds/a685001.html",
        summary_content=(
            "Amoxicillin is a penicillin-class antibiotic that stops the growth of bacteria. "
            "It is prescribed to treat bacterial infections including ear, nose, throat, respiratory tract, "
            "and skin infections. It does NOT treat viral infections such as the common cold or flu. "
            "It is critical to complete the full prescribed course even if symptoms resolve early."
        ),
        key_points=[
            "Antibacterial medication effective only against bacterial organisms",
            "Must be completed for the entire prescribed duration to prevent resistance",
            "Common side effects include mild stomach upset or diarrhea",
        ],
        suggested_doctor_questions=[
            "Should I take this medication with food to prevent stomach irritation?",
            "What should I do if I accidentally miss a scheduled dose?",
        ],
    ),
    "metformin": TrustedSource(
        topic="Metformin",
        source_name="MedlinePlus Drug Information / FDA Prescribing Information",
        source_url="https://medlineplus.gov/druginfo/meds/a681028.html",
        summary_content=(
            "Metformin is a first-line oral biguanide medication for managing type 2 diabetes. "
            "It works by decreasing glucose production in the liver, decreasing glucose absorption in the intestines, "
            "and improving insulin sensitivity. It is commonly taken with meals to reduce gastrointestinal side effects."
        ),
        key_points=[
            "First-line agent for blood glucose control in type 2 diabetes",
            "Improves the body's sensitivity to natural insulin",
            "Best taken with meals to minimize stomach upset",
        ],
        suggested_doctor_questions=[
            "How often should I monitor my blood sugar while taking this medication?",
            "What gastrointestinal side effects are normal, and when should I call your office?",
        ],
    ),
    "lisinopril": TrustedSource(
        topic="Lisinopril",
        source_name="MedlinePlus Drug Information / FDA Prescribing Information",
        source_url="https://medlineplus.gov/druginfo/meds/a692011.html",
        summary_content=(
            "Lisinopril is an angiotensin-converting enzyme (ACE) inhibitor used to lower high blood pressure "
            "and improve survival after heart attacks. By relaxing blood vessels, it makes it easier for the heart "
            "to pump blood. Common mild side effects include a dry, persistent cough or mild dizziness upon standing."
        ),
        key_points=[
            "Relaxes blood vessels to lower blood pressure and protect heart function",
            "Helps prevent long-term kidney complications in hypertensive patients",
            "May occasionally cause a dry cough; do not discontinue without doctor guidance",
        ],
        suggested_doctor_questions=[
            "What target blood pressure reading are we aiming for?",
            "How should I monitor my blood pressure at home while on this medication?",
        ],
    ),
    "atorvastatin": TrustedSource(
        topic="Atorvastatin",
        source_name="MedlinePlus Drug Information / FDA Prescribing Information",
        source_url="https://medlineplus.gov/druginfo/meds/a600045.html",
        summary_content=(
            "Atorvastatin is a statin medication that lowers 'bad' low-density lipoprotein (LDL) cholesterol "
            "and triglycerides in the blood, while raising 'good' HDL cholesterol. By slowing cholesterol production "
            "in the liver, it helps reduce the risk of heart attacks and strokes."
        ),
        key_points=[
            "Lowers LDL cholesterol and cardiovascular event risk",
            "Usually taken once daily, with or without food, often in the evening",
            "Notify your doctor if you experience unexplained muscle pain or tenderness",
        ],
        suggested_doctor_questions=[
            "When should we schedule a follow-up lipid panel to measure the effect?",
            "Are there any specific dietary or grapefruit interactions I need to avoid?",
        ],
    ),
}


def retrieve_trusted_source(name: str) -> Optional[TrustedSource]:
    """Retrieves authoritative medical documentation for a lab test or drug.

    Uses normalized exact matching and fuzzy token similarity.
    Returns None if no trusted source matches above the confidence threshold.
    """
    cleaned = name.lower().strip()
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return None

    # 1. Exact key match
    if cleaned in TRUSTED_MEDICAL_SOURCES:
        return TRUSTED_MEDICAL_SOURCES[cleaned]

    # 2. Substring match against source topic or keys
    for key, source in TRUSTED_MEDICAL_SOURCES.items():
        if key in cleaned or cleaned in key:
            return source
        if source.topic.lower() in cleaned or cleaned in source.topic.lower():
            return source

    # 3. Token-level overlap
    words = cleaned.split()
    for word in words:
        if len(word) >= 4:
            for key, source in TRUSTED_MEDICAL_SOURCES.items():
                if word == key or word in key:
                    return source

    # 4. Fuzzy distance matching (threshold 0.70)
    best_score = 0.0
    best_match = None
    for key, source in TRUSTED_MEDICAL_SOURCES.items():
        ratio = difflib.SequenceMatcher(None, cleaned, key).ratio()
        if ratio > best_score:
            best_score = ratio
            best_match = source

    if best_score >= 0.70:
        return best_match

    return None
