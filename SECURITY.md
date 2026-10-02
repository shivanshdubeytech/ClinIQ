# ClinIQ Security & Data Handling Policy

This document provides a factual assessment of ClinIQ's security posture, data flows, persistence policies, and known architectural boundaries.

---

## 1. External Data Disclosures (What Leaves the System)

Every query and report analysis processes medical information. The following external disclosures occur during normal operation:

- **Groq Cloud API**:
  - **Data Sent**: User question text, retrieved knowledge chunks from ChromaDB, conversation history snippets, and draft medical answers.
  - **Purpose**: Fast inference for answer generation (`generate_answer`) and clinical self-evaluation (`evaluate_and_respond`).
  - **Prerequisite**: Deployment requires an executed zero-data-retention / BAA / DPA with Groq before real patient exposure (see `DEPLOYMENT_BLOCKERS.md` §1.1).
- **Google Text-to-Speech (`gTTS`)**:
  - **Data Sent**: Synthesized answer text (truncated to a maximum of 500 characters).
  - **Purpose**: Voice synthesis for audio response playback (`src/tts.py`).
  - **Prerequisite**: Real patient exposure requires a signed DPA with Google or migration to an on-premises TTS engine.

**No other external telemetry or third-party analytics scripts are included.** All frontend libraries and font assets are self-hosted.

---

## 2. Data Storage & Retention Policies

- **Conversation Memory & Chat History (`cliniq.db`)**:
  - **Stored Data**: User ID, Session ID, input question, assistant answer, confidence score, and creation timestamp.
  - **Retention**: Indefinite. User-facing or automated per-session deletion is not yet implemented.
  - **Database Mode**: SQLite in WAL (Write-Ahead Logging) mode with foreign keys enabled and busy timeout configured to 15 seconds.
- **Synthesized Audio Files (`static/audio/`)**:
  - **Stored Data**: MP3 audio files synthesized for user answers.
  - **Naming**: Cryptographically unguessable random tokens generated via `secrets.token_urlsafe(16)` (128 bits of entropy) matching `^audio_[A-Za-z0-9_-]{1,80}\.mp3$`.
  - **Retention**: Ephemeral. Cleaned automatically by `prune_audio_dir()`: files older than 24 hours (`MAX_AUDIO_AGE_SECONDS = 86400`) are pruned on synthesis and server startup; directory is bounded to 500 files (`MAX_AUDIO_FILES = 500`).
- **Temporary Uploads (`cliniq_uploads/`)**:
  - **Stored Data**: Uploaded document files (PDFs, images).
  - **Path**: UUID-based temporary path under the system temporary directory with `0700` permissions.
  - **Retention**: Deleted immediately in `finally` blocks via `temp_path.unlink()`.
  - **Limitation**: Filesystem unlinking removes directory pointers but does not guarantee cryptographic block zeroization.

---

## 3. Rate Limiting & Boundaries

Process-local sliding-window rate limiting is implemented via `SlidingWindowLimiter` in `app.py`:
- **Write Endpoints**:
  - `/api/query`: 20 requests per 60-second window per client IP.
  - `/api/analyze-report`: 5 requests per 300-second window per client IP.
- **Read Endpoints**:
  - `/api/sessions/<user_id>`: 30 requests per 60-second window per client IP.
  - `/api/session/<session_id>/messages`: 30 requests per 60-second window per client IP.

> [!WARNING]
> **RATE LIMITING IS NOT ACCESS CONTROL.**
> Rate limits slow brute-force abuse and protect provider quotas. They do not authenticate clients and do not prevent data access across shared NAT gateways.

---

## 4. Known Architectural Security Gaps (Out of Scope for Beta)

- **No Authentication**: The application contains no user accounts, passwords, MFA, JWTs, or session cookies.
- **No Per-Record Authorization**: Access to session histories relies solely on the client-provided `user_id` and `session_id`.
- **Public Audio Addressability**: Audio files are addressable by URL to anyone possessing the unguessable 128-bit token; the token functions as a bearer capability.
- **Single-Worker State**: Rate limiter state and SQLite concurrency are bounded to single-worker deployment.

---

## 5. Reporting a Security Vulnerability

If you discover a security vulnerability or sensitive data exposure issue in ClinIQ, please report it privately:
- **Contact**: Email the maintainers directly at `shivanshdubey.tech@gmail.com` with the subject `[ClinIQ Security Advisory]`.
- **Details**: Include reproduction steps, environment details, and affected component paths.
- **Disclosure**: Please allow a 48-hour response window before public coordination.
