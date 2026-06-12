---
id: F015f
title: Containerized deployment + MCP registration/reuse into CoScientist's live catalog
type: feature
status: proposed
created: 2026-06-11
updated: 2026-06-11
owners: [SoloWayG]
derives_from: [F015, F015e]
depends_on: [F015e, F013]
sources: [S027, S028, S029]
tags: [deployment, registration, reuse, mcp, security, sandbox]
code:
  - CoScientist/src/alembic/start_chain.py    # serve_image prints /mcp URL — the seam where registration hooks in
benchmarks: []
---

## Goal
Close the **missing seam**: after Alembic (F015e) builds+commits+serves a tool's MCP server,
**register** that URL into CoScientist's live MCP catalog (the same index F015c queries) and
make it **reusable**, sandboxed and untrusted-by-construction. (build+commit+serve already
exist in `start_chain.py`; this feature is scoped to **register + reuse + sandbox** — see F015 gap.)

## Best practices to adopt
- **ScaleMCP — CRUD auto-sync into the index F015c queries [S027]:** after `serve_image` prints
  the `/mcp` URL, **synchronously upsert** it into the exact index F015c queries (TDWA-weighted
  embedding over tool name + synthetic questions) BEFORE the loop resumes — else a just-built
  tool looks "missing" and re-triggers Alembic. Add a **reuse-lookup** short-circuit for an
  existing `alembic-tool:<repo>` image; dedup new tools vs existing (CRAFT/TroVE).
- **AutoMCP + LATM — verify-before-register [S028]:** mark a tool registered/reusable only after
  4 gates — appears in manifest AND loads AND executes AND performs the expected op; failure
  checklist: missing auth/env creds, unresolved config paths, undeclared params, schema↔signature
  type mismatch. LATM maker/user split: build with a strong pinned model, RUN with a cheap one (cost).
  **⚠ Dedup:** gates 1–3 are largely ALREADY covered by Alembic's validator
  (`validate_syntax`/`run_tests`/`invoke_mcp_tool` → `validation.md`) — at build time, in the
  build container. The net-new part is re-verifying **at registration time against the SERVED
  container** (one real call through the live `/mcp` URL) + recording the readiness level into
  the index entry F015c reads. Also already present on the branch: `start_chain.py:_image_exists`
  (L55) — the image-existence primitive for the reuse-lookup; wire it, don't rewrite it.
- **MCP-security — untrusted-by-construction [S029]:** every Alembic-built server wraps arbitrary
  internet code → own container (done), default-deny egress + per-tool allowlist, CPU/mem/disk
  limits, non-root, secrets via gateway (not baked in), central call logging, localhost+token.
  Port allocator + container GC (else 20000–30000 ports leak).

## ⚠ Risks / open questions (incl. adversarial review)
- **Consistency window:** registration MUST be synchronous (commit before F015c re-queries) or a
  build storm results.
- **Default-deny egress breaks networked tools** (repos that download weights/datasets at runtime)
  → per-tool egress allowlist, not blanket.
- **MCP supply-chain / prompt-injection (missing in first cut):** third-party tool **descriptions
  and outputs** are untrusted inputs that flow into the planner AND into FEDOT.MAS's routing
  meta-agent (which routes purely on server descriptions). Treat as tool-poisoning / tool-shadowing
  / rug-pull surfaces: validate returned content, pin/verify third-party descriptions at registration.

## ✅ TODO
- [ ] Registration seam in `start_chain.serve_image` → synchronous CRUD upsert into the F015c index.
- [ ] Reuse-lookup short-circuit + dedup; AutoMCP 4-gate verify-before-register.
- [ ] Sandbox policy (egress allowlist, non-root, limits, secrets gateway) + port allocator/GC.

## Symbols
- `CoScientist/src/alembic/start_chain.py` — `serve_image` (the registration hook point).
