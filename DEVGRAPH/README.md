# DEVGRAPH — Development & Research Graph

> A persistent, machine-and-human-readable **meta-model of this repository**.
> Where `git` records *what the code is now*, DEVGRAPH records *why it is that way,
> what was tried, what failed, what we still owe, and which ideas (papers / repos)
> each piece of work is grounded in* — so that any agent (or human) resuming work
> can recover project state, known problems, and dead ends **without re-deriving
> them from scratch**.

This file is the **spec**. Read it once to learn the format; then operate via the
short protocol in [§7 Operating protocol](#7-operating-protocol).

---

## 0. TL;DR for an agent (read this first, every session)

```
BOOT  (start of a task)
  1. Read  DEVGRAPH/INDEX.md            ← cheap; the whole map on one screen
  2. Read  DEVGRAPH/project_card.md     ← what the system can do right now
  3. Read  DEVGRAPH/features/<the feature(s) you will touch>.md
           + follow `derives_from` / `depends_on` edges one hop
  4. Skim  the "⚠ Pitfalls" of those features so you don't repeat a known failure

WORK
  - Do the task. Keep a scratch list of: attempts made, what worked/failed,
    which papers/repos you actually used, which symbols you added/changed.

COMMIT-TO-GRAPH  (end of a task — REQUIRED, this is the point of DEVGRAPH)
  5. Append an Attempt to the feature file (method, result, evidence, outcome).
  6. Update the feature `status`, `## ✅ TODO`, `## ⚠ Pitfalls`, `## Symbols`.
  7. Update source `trust` if an idea proved/failed; add new sources you used.
  8. Update INDEX.md row (status, updated date, one-line "now").
  9. Keep claims evidence-backed (see §6 Anti-entrenchment rule).
```

If you do nothing else, do step 5–8. An un-recorded attempt is a failure the next
agent will repeat.

---

## 1. Why this exists (design rationale)

Coding agents are stateless across sessions. The repo's own history (`git log`,
diffs, code comments) tells you *what* changed but is a poor medium for:

- **failed / rejected attempts** — they get squashed, reverted, or never committed,
  so the next agent rediscovers the same dead end;
- **the *why*** — the rationale, the alternative that was considered and dropped;
- **provenance** — which paper/repo an idea came from, and whether that idea
  actually held up when implemented;
- **debt & problems** — TODOs, flaky areas, known pitfalls;
- **capability state** — what the system can do *today*, vs. aspirations.

DEVGRAPH is a small, file-based **knowledge graph** that captures exactly these.
The design borrows from established work (see [§9 Prior art](#9-prior-art--what-this-borrows)):

- **A-MEM (NeurIPS 2025)** — atomic, linked "memory notes" with structured
  attributes (description, keywords, tags) and bidirectional links that *evolve*.
- **Architecture Decision Records (ADRs)** — status lifecycle
  (proposed → accepted → superseded/deprecated/rejected) + context/decision/consequences.
- **Reflexion & the "self-reflection can entrench mistakes" caveat** — hence the
  evidence requirement before anything is marked `rejected`/`refuted` (§6).
- **Voyager skill library** — features point at the *executable* symbols they add,
  not just prose.
- **Codebase knowledge graphs (CONTAIN/CALL/IMPORT/EXTEND edges)** — features are
  *grounded* in real code paths/symbols, so the graph stays falsifiable.
- **A2A Agent Card** — the `project_card.md` capability descriptor.

---

## 2. The graph model

DEVGRAPH is a directed, typed graph stored as Markdown files with YAML
frontmatter. **Nodes are files (or addressable sub-sections); edges are typed
references between IDs.**

### Node types

| Node | ID form | Lives in | What it is |
|------|---------|----------|------------|
| **Project Card** | `project_card` (singleton) | `project_card.md` | Current capabilities, benchmarks, MCP servers, tools. The "what can the system do now" snapshot. |
| **Feature** | `F001`, `F002`, … | `features/F001-<slug>.md` | A unit of work or capability being built/changed. Carries attempts, TODOs, pitfalls, symbols, sources. |
| **Attempt** | `F001.A1`, `F001.A2`, … | a section *inside* a feature file | One concrete try at (part of) a feature: method, result, evidence, outcome. |
| **Decision** | `F001.D1`, … | a section inside a feature file | A mini-ADR: a choice made, alternatives, consequences. (Optional.) |
| **Source** | `S001`, `S002`, … | `sources/S001-<slug>.md` (or a row in `sources/INDEX.md`) | A citation/inspiration: paper, repo, blog, or internal doc, with the *idea extracted* and a *trust* value. |
| **Code Symbol** | `path/to/file.py:Symbol` | referenced inline, not a file | A function/class the work implements. Grounds the feature in real code. |

> We **don't** make a separate file per code symbol (that's the AST's job and it
> rots fast). Symbols are referenced as `path:Symbol` so they're clickable and
> cheaply re-verifiable.

### Edge types

Edges are frontmatter keys (lists of IDs) or inline links. Keep them sparse and
meaningful.

| Edge | From → To | Frontmatter key | Meaning |
|------|-----------|-----------------|---------|
| **derives_from** | Feature → Feature | `derives_from:` | "this work builds on / extends that feature" (the inheritance you asked for). |
| **depends_on** | Feature → Feature | `depends_on:` | needs that feature to function; blocks if it's broken. |
| **supersedes / superseded_by** | Feature → Feature | `supersedes:` / `superseded_by:` | a redesign replacing an older approach. |
| **sources (cites / inspired_by)** | Feature/Attempt → Source | `sources:` | the idea came from here. |
| **refutes** | Attempt → Source | inline in the attempt | "we implemented this source's idea and it did **not** work." Flip the source's `trust` to `refuted`. |
| **implements** | Feature → Code Symbol | `code:` + `## Symbols` | the functions/classes this feature adds/owns. |
| **relates_to** | any → any | `relates_to:` | generic associative link (A-MEM style). Use sparingly. |

Edges should be **bidirectional in spirit**: if `F002.derives_from = [F001]`, then
`F001` should list `F002` under "Descendants" (or you keep it one-way and rely on
`grep`). Default: store the edge once on the *newer* node and let `grep -r "F001"`
find back-references. Don't hand-maintain both directions unless cheap.

### Controlled vocabularies (enums)

**Feature `status`** (maps to your TODO / CLOSE / REJECT):

| status | meaning | your label |
|--------|---------|-----------|
| `proposed` | scoped, not started | — |
| `in_progress` | being built; has open TODOs | **TODO** |
| `blocked` | can't proceed (see `depends_on` / pitfalls) | TODO (stuck) |
| `done` | shipped & verified; TODOs empty | **CLOSE** |
| `rejected` | decided *not* to do it — **requires reason + evidence** | **REJECT** |
| `superseded` | replaced by another feature (`superseded_by`) | — |

**Attempt `outcome`:** `success` · `partial` · `failed` · `abandoned`.

**Source `trust`** (this is the "truth value that can change" you described):

| trust | meaning |
|-------|---------|
| `verified` | we implemented the idea here and it worked. |
| `partial` | worked with caveats / only part of it held. |
| `refuted` | we tried it and it did **not** work for us (record *how* you know). |
| `unverified` | cited / read but not yet empirically tested in this repo. |
| `inspirational` | idea-level influence; not directly testable as-is. |

---

## 3. File layout

```
DEVGRAPH/
├── README.md            ← this spec (how to read & write the graph)
├── INDEX.md             ← FAST BOOT: one-screen map of every feature + status. Read first.
├── project_card.md      ← A2A-style capability card (what the system can do now)
├── features/
│   ├── F001-hitl.md
│   ├── F002-coder-agent.md
│   └── …                ← one file per feature node
└── sources/
    ├── INDEX.md          ← source registry (id · type · trust · url · used-by)
    ├── S001-a-mem.md
    └── …                ← full notes for sources worth more than a registry row
```

Rule of thumb: a source gets its **own file** once it has a non-trivial extracted
idea, a trust history, or is used by >1 feature; otherwise a **row in
`sources/INDEX.md`** is enough.

---

## 4. Node templates

### 4.1 Feature node (`features/F0NN-slug.md`)

```markdown
---
id: F0NN
title: <human title>
type: feature
status: in_progress          # proposed|in_progress|blocked|done|rejected|superseded
created: YYYY-MM-DD
updated: YYYY-MM-DD
owners: [git-handle]
derives_from: []             # [F0xx] features this builds on (inheritance)
depends_on: []               # [F0xx] features required to work
supersedes: []               # [F0xx] / superseded_by: [F0xx]
sources: []                  # [S0xx] inspirations/citations actually used
tags: []                     # free tags for grep/recall
code:                        # implements-edges: paths/symbols this feature owns
  - path/to/file.py:Symbol
benchmarks: []               # results, if any — name + number + link/commit
---

## Goal
<1–3 sentences: what capability/change this delivers and why.>

## Current state
<Where it stands right now in plain language. The "if you read one paragraph" summary.>

## Baseline (only if derives_from is empty / predates the graph)
<The pre-existing functions/classes this work starts from, and what they did,
so the graph is self-contained even without prior recorded nodes.>

## Attempts
### F0NN.A1 — <short name>  ·  YYYY-MM-DD  ·  outcome: success|partial|failed|abandoned
- **Method:** what was tried (the approach/idea, not just the diff).
- **Result:** what happened.
- **Evidence:** how you know — test/CLI output, commit hash, log path, metric. (REQUIRED for failed/abandoned.)
- **Sources used:** [S0xx] …  (mark `refutes S0xx` if a source's idea failed here)
- **Next:** what this implies for the next attempt.

## ✅ TODO
- [ ] open item …            # empty list ⇒ candidate for status: done

## ⚠ Pitfalls / Known problems
- <a trap the next agent must not re-hit, with the reason>.

## Decisions  (optional, mini-ADR)
### F0NN.D1 — <decision>  ·  YYYY-MM-DD
- **Context / Options / Choice / Consequence.**

## Symbols
- `path/to/file.py:Thing` — one line on what it is.
```

### 4.2 Source node (`sources/S0NN-slug.md`)

```markdown
---
id: S0NN
type: paper                  # paper|repo|blog|doc|internal
title: <title>
url: <link>
venue: <e.g. NeurIPS 2025 / GitHub / arXiv id>     # optional
trust: unverified            # verified|partial|refuted|unverified|inspirational
used_by: [F0xx]              # back-edges (optional; grep finds these too)
tags: []
---

## Idea extracted
<The specific claim/technique we took from this source — in our terms.>

## How we used it
<Where it was applied (feature/attempt), and what we changed because of it.>

## Verification log
- YYYY-MM-DD — unverified → <new trust>: <why> (link to attempt F0xx.Ay).
```

---

## 5. Conventions

- **IDs are immutable.** Once `F007` exists, never renumber it; mark `rejected`/
  `superseded` instead. Filenames may change slug; IDs may not.
- **Dates are absolute** (`2026-06-11`), never "today/last week".
- **Grounding over prose.** Every feature should name ≥1 real `code:` path (or be
  `proposed`). A feature claiming behavior with no code reference is suspect.
- **One fact, one place.** Don't duplicate the project card inside features; link.
- **Dedup before create.** Before adding a feature/source, `grep` INDEX for an
  existing one and extend it instead.
- **Stale-reference rule.** Code paths in old nodes rot. *Before* trusting a
  node's `code:`/`Symbols`, verify the symbol still exists (`grep`); if it moved,
  fix the reference as part of your update.

---

## 6. Anti-entrenchment rule (important)

Reflection-style memory has a known failure mode: an agent declares an approach
"doesn't work" on thin evidence, and that false generalization gets frozen into
memory and blocks the right solution forever.

Therefore:

> **A `rejected` feature, a `failed` attempt, or a `refuted` source MUST carry
> concrete evidence** — the command/test that was run, the observed output, a
> commit/log reference, or a specific error. "Seemed not to work" is not enough.

If you only *suspect* something fails, record it as an attempt with
`outcome: partial` and a TODO to confirm — do **not** reject the whole idea.
Refuting a source flips its `trust` to `refuted` *with* a `Verification log`
entry; a future agent may legitimately re-open it with new evidence.

---

## 7. Operating protocol

### On BOOT (resuming/starting work)
1. Read `INDEX.md` → get the map (features, statuses, what's hot, what's blocked).
2. Read `project_card.md` → current capabilities (don't propose things that exist).
3. Open the feature(s) you'll touch; follow `derives_from`/`depends_on` one hop.
4. Read their `## ⚠ Pitfalls` and recent `## Attempts` — **avoid known dead ends.**

### On COMMIT-TO-GRAPH (finishing work) — required
5. **Append an attempt** (`F0NN.A<k>`) with method/result/evidence/outcome.
6. **Update `status`**, tick/strike `## ✅ TODO`, add any `## ⚠ Pitfalls`.
7. **Update `## Symbols`/`code:`** with functions/classes you added or changed.
8. **Update sources:** add ones you actually used; change `trust` for ideas that
   proved or failed (with a Verification-log line); add `refutes` where relevant.
9. **Update `INDEX.md`** row: status, `updated`, and a one-line "now".
10. If you changed system capability, **update `project_card.md`**.

> Scope discipline: only touch nodes for work you actually did this session. Don't
> mass-edit. The graph is a ledger, not a wiki to tidy.

---

## 8. Worked example (how the pieces connect)

`F002-coder-agent` was added (`derives_from: [F000-baseline-orchestrator]`,
`sources: [S002-voyager]`). It implements `CoScientist/tools/coder_tools.py:CoderToolset`
and is registered in `CoScientist/agents/catalog.py`. Its single attempt
`F002.A1` is `outcome: success` with evidence = commit `f863802`. Because the
sandbox idea from Voyager (S002) actually shipped, `S002.trust = partial`
(skill-library-as-code applied; auto-curriculum not). The INDEX row reads:
`F002 · CoderAgent · done · 2026-06-11 · "sandbox coder shipped; no eval yet"`.

See `features/F001-hitl.md` and `features/F002-coder-agent.md` for the real seeded
nodes.

---

## 9. Prior art — what this borrows

The schema is deliberately a *pragmatic distillation* of these (search-verified):

- **A-MEM: Agentic Memory for LLM Agents**, NeurIPS 2025 — Zettelkasten-style
  atomic linked notes with descriptors + evolving links. arXiv:2502.12110.
- **AriGraph** — KG world-models with episodic+semantic memory for LLM agents.
  arXiv:2407.04363.
- **From Experience to Strategy: Trainable Graph Memory** — distilling
  trajectories/attempts into reusable strategy. arXiv:2511.07800.
- **Reflexion** — verbal self-reflection on success/failure; plus the survey
  caveat that reflection can entrench false conclusions (→ §6).
- **Voyager** — skill library of executable code retrieved by description.
  arXiv:2305.16291.
- **Codebase-Memory / KG-over-codebase** — CONTAIN/CALL/IMPORT/EXTEND code graphs
  for agents at far lower token cost. arXiv:2603.27277, arXiv:2505.14394.
- **A2A Agent Card** — capability/skill descriptor → `project_card.md`.
  https://a2a-protocol.org
- **Architecture Decision Records** — status lifecycle + rationale capture.
  https://adr.github.io
- **Google "AI co-scientist"** — generate→critique→rank→evolve research loop, the
  domain this repo lives in.

Full notes (and current trust) for the ones we actually use live in `sources/`.
