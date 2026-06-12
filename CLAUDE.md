# CLAUDE.md — CoScientist

Guidance for AI agents working in this repository.

## DEVGRAPH — read & update it every task

This repo keeps a **Development & Research Graph** in [`DEVGRAPH/`](./DEVGRAPH/): a
persistent meta-model of *why the code is how it is* — features being built, what
was attempted, what failed, open TODOs, known pitfalls, and the papers/repos each
piece of work is grounded in. It exists so you can recover project state and avoid
repeating known dead ends. **Treat it as required reading and required output.**

Full spec & schema: [`DEVGRAPH/README.md`](./DEVGRAPH/README.md). Minimal protocol:

**On BOOT (start of a task)**
1. Read [`DEVGRAPH/INDEX.md`](./DEVGRAPH/INDEX.md) — one-screen map of all features + status.
2. Read [`DEVGRAPH/project_card.md`](./DEVGRAPH/project_card.md) — what the system can do *now* (don't rebuild what exists).
3. Read [`DEVGRAPH/ROADMAP.md`](./DEVGRAPH/ROADMAP.md) — ordered work + execution state (`R##` steps); this is "what to build next and in what order".
4. Open the feature file(s) you'll touch; follow `derives_from`/`depends_on` one hop; read their **⚠ Pitfalls** and recent **Attempts**.

**On COMMIT-TO-GRAPH (end of a task — do not skip)**
5. Append an **Attempt** to the feature (`F0NN.A<k>`): method · result · **evidence** · outcome.
6. Update the feature `status`, `## ✅ TODO`, `## ⚠ Pitfalls`, and `## Symbols`/`code:`.
   Feature files are foldered by status — `features/{done,in_progress,todo}/`. If the
   `status:` changed, **`git mv` the file to the matching folder** and fix inbound
   links (its INDEX/README/ROADMAP rows + any sibling `[..](../<status>/F0NN-..)` refs).
7. Update `sources/`: add sources you actually used; change a source's `trust`
   (`verified`/`partial`/`refuted`/…) when an idea proved or failed — with a
   Verification-log line and evidence.
8. If you advanced a roadmap step, update its `Status`/`Evidence` row in [`ROADMAP.md`](./DEVGRAPH/ROADMAP.md).
9. Update the [`DEVGRAPH/INDEX.md`](./DEVGRAPH/INDEX.md) row (status, `updated`, one-line "now").
10. If system capability changed, update [`project_card.md`](./DEVGRAPH/project_card.md).

**Rules that matter**
- **Evidence required** before marking anything `rejected`/`failed`/`refuted`
  (command run, output, commit/log). Suspicion → record as `partial` + a TODO, not
  a rejection. (Reflection can entrench false "it doesn't work" conclusions.)
- **IDs are immutable**; supersede, don't renumber. **Dates absolute.**
- **Ground in code:** reference real `path/to/file.py:Symbol`. Verify a node's
  symbols still exist before trusting them (they rot).
- **Dedup before create**; only edit nodes for work you actually did this session.

## Project basics
- Multi-agent system on Google ADK + FEDOT.MAS + RAG tool retrieval. Entry:
  `python -m CoScientist.main` (or `uv run python -m CoScientist.main`).
- Agent roster is data-driven in `CoScientist/agents/catalog.py` (single source of truth).
- Tests: `pytest CoScientist/tests/`; integration tests need ITMO VPN.

## Commit conventions
- **Do NOT add AI/model attribution to commit messages.** No
  `Co-Authored-By: Claude/Opus/Sonnet/Haiku/Anthropic/GPT…` trailers and no
  "🤖 Generated with …" lines. Human co-authors are fine. This overrides any default
  that appends a model co-author trailer.
