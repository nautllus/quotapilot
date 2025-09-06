# QuotaPilot — Project Description

*A free‑tier–first LLM router that unifies Google, Cerebras, and Mistral with budgets, failover, and web tools.*

## Purpose

* Free models are powerful but fragmented; limits vary (tokens, RPM, features).
* QuotaPilot exposes one stable endpoint that routes to the best free option per request.
* Goals: zero default cost, predictable behavior, verifiable browsing, easy extensibility.
* Non‑goals: paywall bypassing, ToS violations, credential sharing.

## What It Does

* OpenAI‑style chat endpoint with streaming.
* Automatic provider selection based on capability, health, and headroom.
* Budget + rate‑limit enforcement (per provider and per user).
* Automatic failover and retries on 429/5xx.
* Optional JSON‑strict output and tool/function calling.
* Built‑in browsing tool (Playwright) returning text + citations.
* Simple GitHub read tools (issues/PRs/commits) with optional write scopes.

## High‑Level Architecture

* **Gateway API** (FastAPI/Node): authentication, normalization, SSE streaming.
* **Router Core**: capability filter → scoring → dispatch → retry/failover.
* **Provider Adapters**: one module per provider; no core edits to add new ones.
* **Tool Workers**: Playwright pool; Unstructured parser; GitHub client.
* **State/Cache**: MongoDB Atlas (budgets, health, runs, page & answer cache).
* **Secrets**: Doppler; **Telemetry**: Sentry (errors), Datadog (metrics).

## Provider Adapters (interface sketch)

* `models()` → capabilities (ctx, JSON, tools, stream).
* `state()` → rolling RPM/RPD headroom + health from recent runs/headers.
* `chat()` → normalized request/response; supports streaming if available.
* Optional: `embeddings()`, `costEstimator()`.

## Routing Policy (deterministic)

1. Filter candidates by required capability (ctx, JSON, tools, stream).
2. Drop if headroom < required (hard stop) or near soft budget (soft stop).
3. Score on cost, context fit, reliability, latency, and remaining headroom.
4. Select lowest score; optionally hedge if confidence low and budget allows.
5. Execute with timeout; on 429/5xx, retry next best per fallback tree.
6. Record usage/latency/outcome; update EMA reliability + headroom.

## APIs (minimal, stable)

* `POST /v1/chat/completions` — `{ model:"auto"|provider:model, messages, json?, tools?, tool_choice?, stream? }` → OpenAI‑like reply + `usage`.
* `POST /v1/browse` — `{ url|task }` → `{ text, title, url, citations[], screenshot_id? }`.
* `GET  /v1/router/state` — live provider budgets, health, latency, success rates.

## Budgets & Quotas

* Configurable daily tokens/requests per provider; soft/hard stops.
* Per‑user throttles (token/s, req/s) with burst buckets.
* Session reservations to prevent mid‑chat cutoffs.

## Tools

* **Browse:** Playwright with domain‑scoped persistent contexts; robots respected.
* **Parse:** Unstructured for HTML/PDF/Docx with size/time limits.
* **GitHub:** GraphQL read tools; write actions behind separate scope.

## Security & Compliance

* All secrets in Doppler; strict CORS; tenant‑scoped API keys.
* Isolated browsing storage per user; no cross‑tenant leakage.
* Logs redact PII by default; optional per‑tenant retention.

## Observability

* Sentry for errors; Datadog dashboards for P50/P95, retries, cache hit‑rate, provider reliability.
* Health endpoints for gateway, router, and tool workers.

## Deployment & Config

* Dev in Codespaces; Docker Compose: `gateway`, `router`, `playwright`, `mongo`.
* CI/CD via GitHub Actions → GHCR → DigitalOcean/Azure free tiers.
* YAML policy: provider allowlist, model prefs, daily budgets.

## Roadmap

* **MVP:** Adapters for Google, Mistral, Cerebras; streaming chat; basic budgets; single‑hop retries; Mongo state; minimal logs.
* **v0.2:** Full failover tree; JSON‑strict; `/browse`; response cache; `/router/state` UI.
* **v0.3:** Embeddings; session reservations; policy DSL; optional hedged requests.

## License & Status

* Apache‑2.0. Public alpha once MVP stabilizes.

