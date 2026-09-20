# ClinIQ Knowledge Base Data Sources

## Primary Dataset: MedQuAD (Medical Question Answering Dataset)

The ClinIQ medical RAG knowledge base incorporates a curated subset of **MedQuAD** (Medical Question Answering Dataset), created by Asma Ben Abacha and Demner-Fushman (2019).

* **Repository**: [https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)
* **Description**: MedQuAD includes 47,457 medical question-answer pairs created from 12 National Institutes of Health (NIH) websites.

### NIH-Affiliated Contributing Sources

The dataset draws upon authoritative, peer-reviewed consumer health resources maintained by various NIH institutes and centers:

1. **MedlinePlus Health Topics & Medical Encyclopedia** (U.S. National Library of Medicine - NLM)
2. **NIDDK** (National Institute of Diabetes and Digestive and Kidney Diseases)
3. **GARD** (Genetic and Rare Diseases Information Center)
4. **NCI** (National Cancer Institute)
5. **NHLBI** (National Heart, Lung, and Blood Institute)
6. **CDC** (Centers for Disease Control and Prevention)
7. **GHR** (Genetics Home Reference)

---

## Secondary Dataset: ClinIQ Verified Seed Data

In addition to MedQuAD, ClinIQ incorporates verified core medical QA seed pairs stored in `data/seed_data.py`. These entries cover foundational topics such as diabetes management, hypertension, sleep hygiene, and preventive care.

---

## Academic & Clinical Disclaimer

> **[IMPORTANT DISCLAIMER]**
> The contents of the ClinIQ knowledge base are compiled strictly for academic, research, and informational demonstration purposes. The medical information provided by this system represents general consumer health knowledge and does **NOT** constitute medical advice, professional clinical diagnosis, or treatment recommendations. Always consult a qualified healthcare professional regarding any medical condition or symptoms.
