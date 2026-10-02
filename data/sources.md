# ClinIQ Knowledge Base Data Sources & Ingestion State

## Current Ingestion State

- **Total Ingested Pairs**: 425 Q&A pairs
- **Total Vector Chunks**: 787 chunks (embedded via `all-MiniLM-L6-v2`)
- **Active Vector Collection**: `medical_knowledge` in ChromaDB (`./chroma_db`)
- **Ingestion Pipeline**: `src/ingest.py` (word-based chunking with size=250, overlap=30)

---

## 1. Verified Seed Data (`data/seed_data.py`)
- **Count**: 25 Q&A pairs across 5 core topics:
  - Hypertension / High Blood Pressure
  - Type 2 Diabetes Management
  - Common Cold, Flu & Viral Infections
  - Headaches & Migraines
  - Sleep Hygiene & Insomnia
- **Role**: High-precision benchmark questions for core consumer health inquiries.

---

## 2. MedQuAD Curated Subset (`data/medquad_subset.json`)
- **Count**: 400 Q&A pairs
- **Source**: MedQuAD (Medical Question Answering Dataset, Ben Abacha & Demner-Fushman, 2019) from National Institutes of Health (NIH) consumer health portals:
  - MedlinePlus Health Topics & Encyclopedia
  - NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)
  - NHLBI (National Heart, Lung, and Blood Institute)
  - CDC (Centers for Disease Control and Prevention)
  - GARD (Genetic and Rare Diseases Information Center)
- **Generator**: `data/load_medquad.py` filters, deduplicates, and validates answers.

---

## Clinical Disclaimer

> **[IMPORTANT DISCLAIMER]**
> The contents of the ClinIQ knowledge base are compiled strictly for informational, educational, and beta demonstration purposes. The medical information provided represents general consumer health knowledge and does **NOT** constitute personalized medical advice, clinical diagnosis, or treatment plans. Always consult a qualified healthcare professional regarding any medical condition or symptoms.
