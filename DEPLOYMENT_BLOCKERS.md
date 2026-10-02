# ClinIQ Deployment Blockers
This application must not be exposed to real patients until every item
below is checked. See PRE_DEPLOYMENT_HARDENING.md §1.

- [ ] 1.1 Provider data-processing agreements with Groq and Google (gTTS).
      Patient question text leaves the system on every query. Until a signed
      no-training / zero-retention agreement exists with each provider, the
      application must not serve real patients.

- [ ] 1.2 Clinician sign-off on STANDARD_REFERENCE_RANGES, the critical_high /
      critical_low thresholds, and the urgency wording in the frontend SOS
      dialog. Do not touch these values.

- [ ] 1.3 India privacy / telemedicine regulatory review. An educational
      disclaimer does not settle regulatory status.

- [ ] 1.4 Hosting region selection and confirmation that no data leaves it
      unexpectedly (log aggregation, CDN, TTS).

- [ ] 1.5 TLS certificate provisioning and reverse-proxy deployment. §10 provides
      a config artifact; provisioning the certificate is human work.

---

## Technical & Operational Follow-Ups (Post-Beta Roadmap)

The following operational and security items are documented limitations of the current single-worker beta architecture and must be addressed prior to multi-worker or general production rollout:

- [ ] **Multi-Worker Rate Limiting**: The current `SlidingWindowLimiter` is process-local. Running with multiple gunicorn workers multiplies the effective rate limit by worker count. Migration to a centralized Redis/Valkey rate limiter is required for multi-worker deployments.
- [ ] **SQLite Concurrency & Multi-Worker Storage**: SQLite in WAL mode with busy timeouts supports single-worker concurrent threads. Multi-worker scaling requires migrating session storage and memory to PostgreSQL or MySQL with connection pooling.
- [ ] **Secure Temp File Erasure**: Temporary uploads are deleted using `temp_path.unlink()`. File unlink does not perform cryptographic erasure of disk sectors. Production requires encrypted RAM-disk storage (`tmpfs`) or cryptographic zeroization of upload blocks.
- [ ] **Audio Public Addressability**: Audio tokens (`secrets.token_urlsafe(16)`) provide 128 bits of entropy making filenames unguessable, but files are publicly addressable if the token is leaked. Production should enforce session-authenticated signed URLs or ephemeral streaming.
- [ ] **Content-Security-Policy Style Nonces**: CSP currently permits `'unsafe-inline'` for `style-src` due to inline `<style>` and element style attributes. Full hardening requires refactoring styles to utility classes and implementing cryptographic nonces.
- [ ] **Automated Headless XSS Test Harness**: DOMPurify sanitization is verified in manual audit and code review, but requires an automated headless browser harness (e.g., Playwright/Cypress) integrated in CI.
- [ ] **Clinical Corpus Coverage**: Corpus expanded to 425 Q&A pairs across NIH MedQuAD and seed health topics (787 ChromaDB chunks). Coverage claim in UI copy is consistent.
- [ ] **Production Benchmark Corpus**: Provisional timeouts (`timeout = 180s`, `proxy_read_timeout = 200s`) were established for the single-worker beta. An end-to-end benchmark on production-representative high-resolution scanned PDFs is required to finalize production timeout budgets.
