# ClinIQ — Medical RAG Q&A System

## Project Overview
ClinIQ is an intelligent, Retrieval-Augmented Generation (RAG) question-answering platform designed to provide evidence-based, medically cautious health information. By retrieving context from curated consumer health sources (such as MedlinePlus) and synthesizing answers using Llama-3 via Groq, ClinIQ helps users explore common medical topics safely. The system emphasizes accuracy, transparent confidence scoring, and responsible disclaimers without attempting definitive medical diagnosis.

---

## Architecture
The system is designed with a modular, multi-tier RAG architecture. *(Current status: Scaffolding complete, pipeline logic pending implementation)*:

* **Retrieval Layer** — ChromaDB vector database powered by `sentence-transformers` (`all-MiniLM-L6-v2`) for semantic similarity search over medical Q&A knowledge bases. *(scaffolding only, logic pending)*
* **Generation Layer** — Groq Cloud API integration using `llama-3.1-8b-instant` with structured prompt templates enforcing clinical caution and fallback behavior for low-confidence queries. *(scaffolding only, logic pending)*
* **Evaluation Layer** — Confidence scoring thresholding (minimum 0.6 score) to filter out out-of-scope or low-relevance queries before response generation. *(scaffolding only, logic pending)*
* **Memory Layer** — SQLite database (`cliniq.db`) for persistent conversation history and session tracking. *(scaffolding only, logic pending)*
* **Text-to-Speech (TTS) Layer** — `gTTS` audio synthesis for converting generated medical answers into downloadable MP3 audio files. *(scaffolding only, logic pending)*
* **User Interface (UI) Layer** — Interactive Streamlit web application providing real-time chat, confidence score visualizations, and audio playback. *(scaffolding only, logic pending)*

---

## Tech Stack

| Component | Library / Tool | Version | Role in Project |
| :--- | :--- | :--- | :--- |
| **Vector Store** | `chromadb` | `0.5.23` | Embeddings storage & vector similarity search |
| **Embeddings** | `sentence-transformers` | `3.3.1` | Local sentence embeddings (`all-MiniLM-L6-v2`) |
| **Web UI** | `streamlit` | `1.42.0` | Interactive web dashboard and user interface |
| **LLM Engine** | `groq` | `0.18.0` | Fast cloud inference for `llama-3.1-8b-instant` |
| **Environment** | `python-dotenv` | `1.0.1` | Secure environment variable configuration |
| **Audio / TTS** | `gTTS` | `2.5.4` | Google Text-to-Speech audio generation |

---

## Setup Instructions

### Prerequisites
* Python 3.9+ (Python 3.9, 3.10, or 3.11 recommended)
* A Groq API key (free tier available at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository & Navigate to Workspace
```bash
git clone https://github.com/your-username/cliniq.git
cd cliniq
```

### 2. Create and Activate Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to create your local `.env` configuration file:

**On Linux / macOS:**
```bash
cp .env.example .env
```

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

Edit `.env` and insert your Groq API Key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 5. Verify Setup
To verify that all configuration settings and seed data are correctly configured:
```bash
python -c "from data.seed_data import MEDICAL_QA; print(f'Loaded {len(MEDICAL_QA)} seed Q&A pairs across {len(set(item[\"topic\"] for item in MEDICAL_QA))} topics.')"
```
