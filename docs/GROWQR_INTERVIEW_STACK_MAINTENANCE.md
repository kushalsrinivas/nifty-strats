# GrowQR Interview Stack: What We Must Maintain on Our End

## Purpose
This document defines the internal requirements to reliably run and maintain the GrowQR Interview service stack (Avatar + TTS + STT + LLM analysis + Video CV) at production quality and predictable cost.

Pricing assumptions in this doc come from your current vendor comparison and must be revalidated on a fixed schedule.

## 1. Stack Components We Own
- Avatar streaming orchestration (HeyGen primary)
- Voice synthesis orchestration (ElevenLabs primary, OpenAI/Azure fallback)
- Speech transcription orchestration (Whisper or GPT-4o mini transcribe)
- LLM scoring and feedback workflow (GPT-4o mini + GPT-4o)
- Video analysis workflow (AWS Rekognition video or sampled image mode)
- Session state, storage, queues, retries, and audit logs

## 2. Mandatory Vendor and Billing Setup
- Active paid accounts with sufficient credits for:
  - HeyGen
  - ElevenLabs
  - OpenAI
  - AWS (Rekognition + storage + logging)
- Auto-recharge enabled where available.
- Billing alerts at 50%, 75%, 90%, and 100% of monthly budget.
- One owner for billing and one backup owner.
- Current rate card snapshot saved monthly in a shared folder.

## 3. Security and Access Requirements
- All API keys stored in a secrets manager, never in source code.
- Key rotation every 90 days (or sooner after incident/offboarding).
- Separate keys per environment (`dev`, `staging`, `prod`).
- Least-privilege IAM for cloud services and CI/CD jobs.
- Mandatory MFA for billing/admin accounts.

## 4. Core Infrastructure We Must Run
- API gateway + backend interview orchestrator service.
- Worker queue for async tasks:
  - transcription post-processing
  - LLM scoring
  - video analysis
- Durable object storage for interview audio/video artifacts.
- Session database for metadata, scores, transcript segments, and status.
- Centralized logs and metrics with alerting.

## 5. Required Runtime Configuration
At minimum, maintain these env vars/secrets:
- `HEYGEN_API_KEY`
- `ELEVENLABS_API_KEY`
- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `REKOGNITION_COLLECTION_ID` (if used)
- `STORAGE_BUCKET_INTERVIEW_MEDIA`
- `LLM_SCORING_MODEL` (for example `gpt-4o-mini`)
- `LLM_FEEDBACK_MODEL` (for example `gpt-4o`)
- `TTS_PROVIDER_PRIMARY` / `TTS_PROVIDER_FALLBACK`
- `STT_PROVIDER_PRIMARY` / `STT_PROVIDER_FALLBACK`
- `COST_PER_SESSION_HARD_LIMIT_USD`

## 6. Reliability and Fallback Policy
- If Avatar fails or exceeds latency threshold:
  - fallback to audio-only interview mode.
- If ElevenLabs fails:
  - fallback to OpenAI TTS.
- If STT fails:
  - retry once, then switch provider.
- If video analysis fails:
  - fallback to sampled-frame image analysis mode.
- Each external call should have:
  - timeout
  - retry with exponential backoff
  - idempotency key
  - structured error code in logs

## 7. Observability and SLOs
Track these metrics per provider and per interview session:
- request count, success rate, error rate
- p50/p95/p99 latency
- retries and fallback rate
- cost per call and cost per completed interview
- token/character/minute usage per component

Minimum target SLOs (starting point):
- Interview session start success >= 99.0%
- End-to-end interview completion success >= 98.0%
- Scoring delivery within 60 seconds after interview end >= 95.0%

## 8. Cost Control Requirements
Cost is dominated by avatar minutes, so enforce guardrails:
- Real-time cost meter per live interview.
- Soft warning when session cost crosses 70% of target.
- Hard stop/degrade mode when session crosses configured limit.
- Daily anomaly alert when:
  - spend/session rises by >20% day-over-day, or
  - fallback rate rises by >10%.

Use this session cost formula:

`session_cost = avatar_cost + tts_cost + stt_cost + llm_cost + video_cv_cost`

For your current assumptions:
- 15-minute baseline is roughly `$4.75/session`.
- Set alert threshold initially around `$5.50/session` until stable.

## 9. Data Governance and Compliance
- Explicit user consent for audio/video capture and AI analysis.
- Data retention policy by artifact type:
  - raw video/audio
  - transcripts
  - scoring JSON
  - analytics events
- Encryption in transit (TLS) and at rest.
- PII access logs and audit trail.
- Deletion workflow for right-to-delete requests.

## 10. QA and Release Process
- Maintain `dev`, `staging`, and `prod` environments.
- Before release:
  - synthetic end-to-end test (avatar -> tts -> stt -> llm -> report)
  - failure-injection test for each provider
  - cost regression check on a fixed test interview set
- Keep golden test sessions to compare score drift after model/version updates.

## 11. Operating Checklist
Daily:
- Check provider health dashboards and failed sessions.
- Review error spikes, fallback spikes, and p95 latency.
- Verify cost/session is within threshold.

Weekly:
- Review provider usage and optimize routing rules.
- Tune model/provider mix for cost vs quality.
- Audit top 10 failed interviews and root causes.

Monthly:
- Revalidate pricing and update internal cost sheet.
- Rotate keys if due.
- Review retention/deletion compliance logs.
- Capacity review for expected interview volume.

## 12. Ownership Model (Minimum)
- Backend owner: orchestration, retries, APIs, queues.
- AI owner: scoring prompts, model routing, quality checks.
- Infra/SRE owner: uptime, alerting, cost controls, incident response.
- Product owner: rubric quality, interview UX, policy decisions.

## 13. Immediate Actions to Stay Safe in Production
1. Implement per-session real-time cost tracking before scale-up.
2. Add hard fallback to audio-only mode when avatar quality drops.
3. Separate scoring and feedback models (`gpt-4o-mini` and `gpt-4o`) with routing rules.
4. Add monthly vendor price review task and update this document.
