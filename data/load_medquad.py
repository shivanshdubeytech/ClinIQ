"""MedQuAD Dataset Loader for ClinIQ medical RAG system.

Downloads/clones MedQuAD dataset, parses XML files, applies quality filters,
deduplicates questions, prioritizes key health topics, and exports clean JSON data.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import urllib.request
import zipfile

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MEDQUAD_REPO_DIR = DATA_DIR / "MedQuAD_repo"
MEDQUAD_ZIP_PATH = DATA_DIR / "medquad.zip"
OUTPUT_JSON_PATH = DATA_DIR / "medquad_subset.json"

MEDQUAD_ZIP_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
MEDQUAD_GIT_URL = "https://github.com/abachaa/MedQuAD.git"

# Priority topics to ensure broad & deep medical coverage
PRIORITY_TOPICS = [
    "diabetes",
    "hypertension",
    "heart disease",
    "cold",
    "flu",
    "influenza",
    "headache",
    "migraine",
    "sleep",
    "insomnia",
    "anxiety",
    "asthma",
    "allergy",
    "allergies",
    "prevention",
    "preventive care",
    "nutrition",
    "exercise",
]

# Invalid / low-quality answer indicators
INVALID_ANSWER_SUBSTRINGS = [
    "no answer available",
    "information is not available",
    "page not found",
    "refer to your physician",
    "refer to a physician",
    "content not available",
    "do not have an answer",
    "not provided",
    "unspecified",
]


def download_or_clone_medquad() -> Path:
    """Clones git repository or downloads zip archive of MedQuAD if not present."""
    if MEDQUAD_REPO_DIR.exists() and any(MEDQUAD_REPO_DIR.iterdir()):
        print(f"[MEDQUAD] Found existing MedQuAD data at '{MEDQUAD_REPO_DIR}'.")
        return MEDQUAD_REPO_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try git clone first
    print(f"[MEDQUAD] Attempting git clone from '{MEDQUAD_GIT_URL}'...")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", MEDQUAD_GIT_URL, str(MEDQUAD_REPO_DIR)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[MEDQUAD] Git clone successful.")
        return MEDQUAD_REPO_DIR
    except Exception as e:
        print(f"[MEDQUAD] Git clone failed ({e}). Falling back to ZIP download...")

    # Fallback to ZIP download
    print(f"[MEDQUAD] Downloading ZIP from '{MEDQUAD_ZIP_URL}'...")
    urllib.request.urlretrieve(MEDQUAD_ZIP_URL, MEDQUAD_ZIP_PATH)
    print(f"[MEDQUAD] Extracting ZIP file...")

    with zipfile.ZipFile(MEDQUAD_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Rename extracted folder if needed (usually MedQuAD-master)
    extracted_dir = DATA_DIR / "MedQuAD-master"
    if extracted_dir.exists():
        if MEDQUAD_REPO_DIR.exists():
            shutil.rmtree(MEDQUAD_REPO_DIR)
        extracted_dir.rename(MEDQUAD_REPO_DIR)

    if MEDQUAD_ZIP_PATH.exists():
        MEDQUAD_ZIP_PATH.unlink()

    print(f"[MEDQUAD] Download and extraction complete.")
    return MEDQUAD_REPO_DIR


def clean_text(text: Optional[str]) -> str:
    """Strips whitespace and normalizes HTML/XML formatting in string."""
    if not text:
        return ""
    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_question(q: str) -> str:
    """Normalizes question string for deduplication."""
    q_clean = re.sub(r"[^\w\s]", "", q.lower())
    return re.sub(r"\s+", " ", q_clean).strip()


def parse_xml_file(filepath: Path) -> List[Dict[str, str]]:
    """Parses a single MedQuAD XML file into list of QA pair dicts.

    Handles malformed XML gracefully by returning empty list on failure.
    """
    qa_list: List[Dict[str, str]] = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception as err:
        print(f"[MEDQUAD] Warning: Skipping malformed XML file '{filepath.name}': {err}")
        return qa_list

    # Extract source from Document tag attribute or parent folder
    source = root.attrib.get("source", "")
    if not source:
        source = filepath.parent.name

    # Extract Focus / Topic
    focus_elem = root.find("Focus")
    focus_text = focus_elem.text if focus_elem is not None else ""
    focus = clean_text(focus_text) or filepath.stem

    # Iterate over QAPair elements
    for qa in root.findall(".//QAPair"):
        question_elem = qa.find("Question")
        answer_elem = qa.find("Answer")

        if question_elem is None or answer_elem is None:
            continue

        question_text = clean_text(question_elem.text)
        answer_text = clean_text(answer_elem.text)

        if question_text and answer_text:
            qa_list.append(
                {
                    "topic": focus,
                    "question": question_text,
                    "answer": answer_text,
                    "source": f"MedQuAD ({source})",
                }
            )

    return qa_list


def is_quality_answer(answer: str) -> bool:
    """Validates answer against quality criteria."""
    if len(answer) < 40:
        return False

    ans_lower = answer.lower()
    for bad_substr in INVALID_ANSWER_SUBSTRINGS:
        if bad_substr in ans_lower:
            return False

    return True


def is_priority_entry(entry: Dict[str, str]) -> bool:
    """Checks if an entry belongs to target priority topics."""
    text_to_check = f"{entry['topic']} {entry['question']}".lower()
    return any(p in text_to_check for p in PRIORITY_TOPICS)


def process_medquad_data(
    target_count: int = 400,
) -> Tuple[List[Dict[str, str]], int, int, int]:
    """Parses MedQuAD dataset, applies quality filters, deduplicates, and caps subset size.

    Returns:
        Tuple: (selected_entries, total_parsed, total_filtered_pass, total_stored)
    """
    repo_path = download_or_clone_medquad()

    # Find all XML files
    xml_files = list(repo_path.rglob("*.xml"))
    print(f"[MEDQUAD] Found {len(xml_files)} XML files across MedQuAD source directories.")

    total_parsed = 0
    total_passed_filters = 0
    seen_questions: Set[str] = set()

    priority_entries: List[Dict[str, str]] = []
    other_entries: List[Dict[str, str]] = []

    for xml_file in xml_files:
        qa_pairs = parse_xml_file(xml_file)
        for entry in qa_pairs:
            total_parsed += 1

            # Quality filtering
            if not is_quality_answer(entry["answer"]):
                continue

            # Question deduplication
            q_norm = normalize_question(entry["question"])
            if not q_norm or q_norm in seen_questions:
                continue

            seen_questions.add(q_norm)
            total_passed_filters += 1

            if is_priority_entry(entry):
                priority_entries.append(entry)
            else:
                other_entries.append(entry)

    print(
        f"[MEDQUAD] Parsing complete. Total parsed: {total_parsed}, "
        f"Passed quality filters & deduplicated: {total_passed_filters} "
        f"(Priority: {len(priority_entries)}, Other: {len(other_entries)})."
    )

    # Select target subset (~300-500 QA pairs, taking priority entries first)
    selected: List[Dict[str, str]] = []
    if len(priority_entries) >= target_count:
        selected = priority_entries[:target_count]
    else:
        selected = priority_entries + other_entries[: (target_count - len(priority_entries))]

    total_stored = len(selected)

    # Save cleaned result to JSON file
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"[MEDQUAD] Saved {total_stored} curated entries to '{OUTPUT_JSON_PATH}'.")
    return selected, total_parsed, total_passed_filters, total_stored


if __name__ == "__main__":
    entries, parsed, passed, stored = process_medquad_data(target_count=400)
    print("\n--- SUMMARY ---")
    print(f"Total Entries Parsed:       {parsed}")
    print(f"Passed Quality Filters:     {passed}")
    print(f"Stored in Subset JSON:      {stored}")
    print(f"Output File:                {OUTPUT_JSON_PATH}")
