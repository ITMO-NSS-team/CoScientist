/**
 * Live status indicator — ONE human sentence about what the system is doing.
 *
 * Why a component of its own: everything needed already arrives over the chat
 * websocket (`status`, `agent_event`, `agent_output`, `tool_activity`,
 * `hitl_request`, `final_response`), but in raw form — tool names, call ids,
 * JSON previews. The activity rail and the ToolsViewer render exactly that,
 * for a developer. A user who is not one needs the opposite: no tool names, no
 * ids, one line that says "I am searching the web" and a way to expand it.
 *
 * So this module is a *reducer*: every websocket frame is fed in, and a small
 * state machine decides which phrase is true right now. Nothing here talks to
 * the server, and nothing on the server knows about it — which is why it also
 * works on snapshot replay (a reconnect, a session switch) and in `?demo=`
 * mode, where a recorded `agent_events.json` is played back with no run at all.
 *
 * Public API (all no-ops before `mount()`):
 *   StatusIndicator.mount(el)          — attach to a container element
 *   StatusIndicator.feed(msg, quiet)   — push one websocket frame
 *   StatusIndicator.setLang('ru'|'en') — phrase language
 *   StatusIndicator.setConnected(bool) — websocket up/down
 *   StatusIndicator.markStopped()      — the user pressed Stop
 *   StatusIndicator.reset()            — session switch / clear
 *   StatusIndicator.demo(file, speed)  — replay a saved bundle's events
 */
(function () {
  'use strict';

  // ── Tuning ────────────────────────────────────────────────────────────────
  // A phrase is held at least this long even if the next tool call arrives
  // sooner — a run that fires five tools in 200 ms must not strobe.
  const MIN_HOLD_MS = 1200;
  const RENDER_THROTTLE_MS = 350;
  // How long silence must last before "working" decays into "thinking".
  const SETTLE_GRACE_MS = 900;
  // No event at all for this long: stop claiming a specific action.
  const SILENCE_MS = 20000;
  const LONG_RUN_MS = 60000;   // add the "this may take a few minutes" note
  const DONE_LINGER_MS = 2600; // green check, then hide
  const ERROR_LINGER_MS = 8000;
  const STOPPED_LINGER_MS = 3000;
  const MAX_STEPS = 6;
  const DETAIL_LIMIT = 64;

  // ── Phrase dictionary ─────────────────────────────────────────────────────
  // Categories are deliberately coarse: the user cares about the KIND of work,
  // not which of four search backends answered.
  const CATEGORIES = {
    web_search: { icon: 'travel_explore', ru: 'Ищу информацию в интернете', en: 'Searching the web' },
    papers: { icon: 'menu_book', ru: 'Читаю научные статьи', en: 'Reading scientific papers' },
    rag: { icon: 'database', ru: 'Ищу в базе знаний', en: 'Searching the knowledge base' },
    code: { icon: 'terminal', ru: 'Пишу и запускаю код', en: 'Writing and running code' },
    data: { icon: 'dataset', ru: 'Готовлю данные', en: 'Preparing the data' },
    chem: { icon: 'science', ru: 'Считаю свойства молекул', en: 'Computing molecular properties' },
    plot: { icon: 'insert_chart', ru: 'Строю графики', en: 'Building charts' },
    graph_write: { icon: 'hub', ru: 'Записываю найденное в граф знаний', en: 'Recording findings in the knowledge graph' },
    graph_read: { icon: 'account_tree', ru: 'Сверяюсь с картой исследования', en: 'Checking the research map' },
    mcp_build: { icon: 'construction', ru: 'Собираю новый инструмент', en: 'Building a new tool' },
    mcp_find: { icon: 'extension', ru: 'Подбираю инструменты для задачи', en: 'Picking tools for the task' },
    tasks: { icon: 'checklist', ru: 'Планирую шаги', en: 'Planning the steps' },
    files: { icon: 'description', ru: 'Работаю с документами', en: 'Working with documents' },
    tool: { icon: 'build', ru: 'Работаю с инструментом', en: 'Using a tool' },
  };

  // First match wins, so the specific patterns come before the generic ones.
  // Order matters more than it looks: `research_*` (the research-graph tools)
  // contains the substring "search", so it has to be decided long before the
  // catch-all search rule at the bottom, or half the run reads as web search.
  const RULES = [
    [/(build|check)_mcp|mcp_build|alembic/, 'mcp_build'],
    [/search_mcp|register_mcp|mcp_server|tool_retriev|rerank|prepare_tool/, 'mcp_find'],
    [/create_plan|task_status|active_tasks|roadmap|todo/, 'tasks'],
    [/chroma|embed|vector|knowledge_memory|memory|recall|retriev/, 'rag'],
    [/research_commit|graph_commit|upsert|add_node|write_graph/, 'graph_write'],
    [/research_|graph|_node|_edge/, 'graph_read'],
    [/arxiv|pubmed|openalex|semantic_scholar|scholar|paper|pico|doi|crossref/, 'papers'],
    [/rdkit|smiles|pubchem|molecul|chembl|inchi|admet|reaction/, 'chem'],
    [/plot|chart|figure|histogram|visuali/, 'plot'],
    [/sandbox|shell|bash|exec|python|jupyter|notebook|git_|npm|pip_|coder/, 'code'],
    [/dataset|csv|parquet|upload|download|fedot|automl|table/, 'data'],
    [/tavily|serp|duckduckgo|google|web_search|browse|fetch_url|crawl/, 'web_search'],
    [/file|read|write|document|report|format_results|pdf|marker|parse/, 'files'],
  ];

  // Deliberately kept out of RULES: "lookup"/"query"/"find" appear in the name
  // of half the tools ever written, so this guess is only worth making after a
  // tool's own description has had its say.
  const GENERIC_RULE = [/search|query|lookup|find|browse/, 'web_search'];

  // Agent roles, as a user would name them. Anything missing falls back to the
  // bare class name with the "Agent" suffix stripped.
  const AGENTS = {
    OrchestratorAgent: { ru: 'Координатор', en: 'Orchestrator' },
    PlannerAgent: { ru: 'Планировщик', en: 'Planner' },
    ContextInitAgent: { ru: 'Уточнение задачи', en: 'Task intake' },
    HypothesesAgent: { ru: 'Генератор гипотез', en: 'Hypotheses' },
    ResearchAgent: { ru: 'Исследователь', en: 'Researcher' },
    TaskExecutorAgent: { ru: 'Исполнитель', en: 'Executor' },
    ToolPipelineAgent: { ru: 'Подбор инструментов', en: 'Tool pipeline' },
    CoderAgent: { ru: 'Инженер', en: 'Engineer' },
    DatasetCollectorAgent: { ru: 'Сбор данных', en: 'Data collector' },
    MedicalAgent: { ru: 'Медицинский аналитик', en: 'Medical analyst' },
    McpBuilderAgent: { ru: 'Сборщик инструментов', en: 'Tool builder' },
    ToolPreparerAgent: { ru: 'Подготовка инструментов', en: 'Tool preparer' },
    // The tool pipeline fans out into half a dozen internal agents. Naming each
    // one tells a user nothing — they are all the same activity to them.
    ParallelToolSearcherAgent: { ru: 'Подбор инструментов', en: 'Tool search' },
    LocalToolsExtractorAgent: { ru: 'Подбор инструментов', en: 'Tool search' },
    ToolRetrieverAgent: { ru: 'Подбор инструментов', en: 'Tool search' },
    ToolWebSearcherAgent: { ru: 'Подбор инструментов', en: 'Tool search' },
    ToolReranker: { ru: 'Подбор инструментов', en: 'Tool search' },
    FullSetToolReranker: { ru: 'Подбор инструментов', en: 'Tool search' },
    WebToolsDeployerAgent: { ru: 'Подключаю инструменты', en: 'Deploying tools' },
    ResultAggregatorAgent: { ru: 'Составитель отчёта', en: 'Report writer' },
    ExperimentAgent: { ru: 'Эксперимент', en: 'Experiment' },
    FedotAgent: { ru: 'AutoML', en: 'AutoML' },
    system: { ru: 'Система', en: 'System' },
  };

  // Phrases that are not about a tool at all.
  const PHASES = {
    starting: { icon: 'bolt', ru: 'Принимаю задачу', en: 'Picking up the task' },
    thinking: { icon: 'neurology', ru: 'Обдумываю', en: 'Thinking it through' },
    long_thinking: { icon: 'neurology', ru: 'Думаю над задачей', en: 'Working on the task' },
    delegating: { icon: 'alt_route', ru: 'Передаю задачу', en: 'Handing over to' },
    finalizing: { icon: 'auto_awesome', ru: 'Собираю итоговый отчёт', en: 'Assembling the final report' },
    waiting: { icon: 'front_hand', ru: 'Жду вашего ответа', en: 'Waiting for your answer' },
    done: { icon: 'check_circle', ru: 'Готово', en: 'Done' },
    error: { icon: 'error', ru: 'Что-то пошло не так', en: 'Something went wrong' },
    stopped: { icon: 'stop_circle', ru: 'Остановлено', en: 'Stopped' },
    offline: { icon: 'cloud_off', ru: 'Соединение потеряно, переподключаюсь', en: 'Connection lost, reconnecting' },
  };

  const TEXT = {
    longRun: { ru: 'это может занять несколько минут', en: 'this may take a few minutes' },
    toolFailed: { ru: 'Инструмент не ответил, пробую иначе', en: 'A tool did not answer — trying another way' },
    step: { ru: 'Шаг %d из %d', en: 'Step %d of %d' },
    details: { ru: 'Подробнее', en: 'Details' },
    ready: { ru: 'Результат готов', en: 'Result ready' },
  };

  // Colour + iconography per phase, in the page's own Tailwind tokens.
  const TONES = {
    work: { bar: 'bg-primary', icon: 'text-primary', card: 'border-primary/25 bg-primary/[0.06]', text: 'text-on-surface' },
    wait: { bar: 'bg-tertiary', icon: 'text-tertiary', card: 'border-tertiary/40 bg-tertiary/[0.08]', text: 'text-on-surface' },
    done: { bar: 'bg-secondary', icon: 'text-secondary', card: 'border-secondary/30 bg-secondary/[0.06]', text: 'text-on-surface' },
    fail: { bar: 'bg-error', icon: 'text-error', card: 'border-error/40 bg-error/[0.08]', text: 'text-on-surface' },
    mute: { bar: 'bg-outline-variant', icon: 'text-outline-variant', card: 'border-outline-variant/25 bg-surface-container-high/40', text: 'text-on-surface-variant' },
  };

  const PHASE_TONE = {
    waiting: 'wait', done: 'done', error: 'fail', stopped: 'mute', offline: 'mute',
  };

  const EXPAND_KEY = 'coscientist.status_expanded';

  // ── State ─────────────────────────────────────────────────────────────────
  let lang = 'ru';
  let root = null;          // container element
  let expanded = false;
  let connected = true;

  const st = newState();

  function newState() {
    return {
      phase: 'idle',
      category: null,
      agent: null,
      detail: null,
      target: null,        // agent a delegation points at
      toolName: null,      // only used by the unrecognised-tool phrase
      since: 0,            // when the current run started
      phraseSince: 0,      // when the visible phrase was last changed
      lastEventAt: 0,
      inflight: new Map(), // call_id -> {tool, agent, category, at}
      anonymous: 0,        // in-flight calls that arrived without a call id
      steps: [],           // [{key, icon, text, status}]
      stepIndex: new Map(),// call_id -> step
      // The orchestrator's plan (the task tracker) and, nested inside one of
      // its steps, the plan the agent in the sandbox keeps for itself. Both
      // are {current, done, total}; both are null until something plans.
      tasks: [],           // task-tracker items, kept so a status update lands
      plan: null,
      sandboxPlan: null,
      note: null,          // transient sub-line (a failed tool, …)
      hideAt: 0,           // when a terminal phase should disappear
    };
  }

  // ── Small helpers ─────────────────────────────────────────────────────────
  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s === null || s === undefined ? '' : String(s);
    return d.innerHTML;
  }

  function pick(entry, fallback) {
    if (!entry) return fallback || '';
    return entry[lang] || entry.en || entry.ru || fallback || '';
  }

  function agentLabel(name) {
    if (!name) return '';
    const known = AGENTS[name];
    if (known) return pick(known, name);
    return String(name).replace(/Agent$/, '') || name;
  }

  /** Which category a tool call belongs to.
   *
   *  The name is tried first and is right for everything the system ships. A
   *  tool from an MCP server built at runtime is named by whatever repository
   *  it came from, though, so an unrecognised name falls back to matching the
   *  same rules against the tool's own description, which `tool_activity`
   *  carries on every call record. */
  function classify(toolName, description) {
    const name = String(toolName || '').toLowerCase();
    const text = String(description || '').toLowerCase();
    const match = (haystack) => {
      for (const [pattern, category] of RULES) {
        if (pattern.test(haystack)) return category;
      }
      return null;
    };
    return match(name)
      || (text && match(text))
      || (GENERIC_RULE[0].test(name) ? GENERIC_RULE[1] : null)
      || (text && GENERIC_RULE[0].test(text) ? GENERIC_RULE[1] : null)
      || 'tool';
  }

  /** A delegation, not tool use: `transfer_to_agent`, or an AgentTool whose
   *  tool name IS the subordinate's agent name. Mirrors the activity rail. */
  function delegationTarget(toolName, args) {
    if (toolName === 'transfer_to_agent') {
      return (args && (args.agent_name || args.agentName)) || null;
    }
    return /Agent$/.test(String(toolName || '')) ? toolName : null;
  }

  /** The one argument worth showing a human — a query, a path, a molecule.
   *
   *  Deliberately strict. `tool_activity` hands over a *string* whenever the
   *  arguments were too big to keep as structure, so a string here is a
   *  truncated JSON dump, never a sentence: showing it would put
   *  `{"nodes": [{"ref": "e1"…` under the phrase. Same for any single value
   *  that is itself serialized structure. */
  function detailOf(args) {
    if (!args || typeof args !== 'object' || Array.isArray(args)) return null;
    const keys = ['query', 'q', 'search_query', 'question', 'task', 'topic',
      'name', 'smiles', 'molecule', 'url', 'path', 'file_path', 'filename',
      'title', 'command', 'request'];
    for (const key of keys) {
      const value = args[key];
      if (typeof value !== 'string') continue;
      const text = value.trim();
      if (!text || /^[[{]/.test(text)) continue;
      return trim(text);
    }
    return null;
  }

  function trim(value) {
    const text = String(value).replace(/\s+/g, ' ').trim();
    return text.length > DETAIL_LIMIT ? text.slice(0, DETAIL_LIMIT - 1) + '…' : text;
  }

  function elapsed(ms) {
    const total = Math.max(0, Math.round(ms / 1000));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return m + ':' + String(s).padStart(2, '0');
  }

  function stamp(timestamp) {
    if (!timestamp) return Date.now();
    const parsed = new Date(timestamp).getTime();
    return Number.isFinite(parsed) ? parsed : Date.now();
  }

  // ── Steps (the expanded view) ─────────────────────────────────────────────
  function pushStep(key, icon, text) {
    const step = { key: key, icon: icon, text: text, status: 'running' };
    st.steps.push(step);
    if (key) st.stepIndex.set(key, step);
    while (st.steps.length > MAX_STEPS) {
      const dropped = st.steps.shift();
      if (dropped.key) st.stepIndex.delete(dropped.key);
    }
    return step;
  }

  function closeStep(key, failed) {
    const step = key ? st.stepIndex.get(key) : null;
    const target = step || [...st.steps].reverse().find(s => s.status === 'running');
    if (target) target.status = failed ? 'error' : 'done';
  }

  // ── Phrase transitions ────────────────────────────────────────────────────
  let holdTimer = null;

  /** Set the visible phase, holding the previous phrase for MIN_HOLD_MS so a
   *  burst of fast tool calls reads as a sequence instead of a flicker.
   *  Terminal phases (done/error/stopped) always land immediately. */
  function setPhase(phase, patch, immediate) {
    const now = Date.now();
    const apply = () => {
      st.phase = phase;
      st.category = (patch && 'category' in patch) ? patch.category : null;
      st.detail = (patch && 'detail' in patch) ? patch.detail : null;
      st.target = (patch && 'target' in patch) ? patch.target : null;
      st.toolName = (patch && 'toolName' in patch) ? patch.toolName : null;
      if (patch && patch.agent) st.agent = patch.agent;
      st.phraseSince = Date.now();
      render();
    };
    if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
    const waited = now - st.phraseSince;
    if (immediate || st.phase === 'idle' || waited >= MIN_HOLD_MS) {
      apply();
    } else {
      holdTimer = setTimeout(apply, MIN_HOLD_MS - waited);
      render(); // keep the timer/sub-line live meanwhile
    }
  }

  /** Whatever is true once no tool call is outstanding.
   *
   *  Agents chain calls back to back — a result and the next call are often
   *  milliseconds apart — so "thinking" is only declared after a short grace
   *  period. Without it the line alternates work/think on every single tool. */
  let settleTimer = null;

  function settle() {
    if (st.phase === 'waiting') return;
    if (st.inflight.size + st.anonymous > 0) return;
    if (settleTimer) clearTimeout(settleTimer);
    const apply = () => {
      settleTimer = null;
      if (st.inflight.size + st.anonymous > 0) return;
      if (st.phase === 'waiting' || st.phase === 'idle') return;
      setPhase(finalStage() ? 'finalizing' : 'thinking', { agent: st.agent });
    };
    if (st.phase === 'starting' || st.phase === 'idle') apply();
    else settleTimer = setTimeout(apply, SETTLE_GRACE_MS);
  }

  function cancelSettle() {
    if (settleTimer) { clearTimeout(settleTimer); settleTimer = null; }
  }

  function finalStage() {
    return st.agent === 'ResultAggregatorAgent';
  }

  function startRun(at) {
    // A second message in the same session starts a new run: the previous
    // run's steps and plan counters must not bleed into it.
    if (['idle', 'done', 'stopped', 'error'].includes(st.phase)) {
      Object.assign(st, newState(), { since: at || Date.now() });
    }
    st.lastEventAt = Date.now();
    st.hideAt = 0;
    setPhase('starting', {}, true);
  }

  function endRun(phase, linger) {
    // A failed run may still be followed by an idle `status` broadcast. It must
    // not turn "something went wrong" into a green "done".
    if (phase === 'done' && st.phase === 'error') return;
    cancelSettle();
    if (phase === 'done') st.note = null;
    st.inflight.clear();
    st.anonymous = 0;
    st.steps.forEach(step => { if (step.status === 'running') step.status = 'done'; });
    st.hideAt = Date.now() + linger;
    setPhase(phase, {}, true);
    setTimeout(() => {
      if (st.hideAt && Date.now() >= st.hideAt - 50) {
        st.phase = 'idle';
        st.hideAt = 0;
        render();
      }
    }, linger + 60);
  }

  // ── Plan progress ─────────────────────────────────────────────────────────
  // Two plans exist at once and they are not the same thing. The orchestrator's
  // task tracker plans the *study*; the agent inside the sandbox plans the job
  // it was handed, which is one step of that study. Both are read here into the
  // same shape, and the sub-line prefers whichever is more specific.

  const DONE_STATUS = /done|complete|finish/i;
  const RUNNING_STATUS = /progress|running|active|doing/i;

  /** {total, done, current} for a list of {title, status} items. */
  function summarise(items) {
    if (!Array.isArray(items) || !items.length) return null;
    const done = items.filter(item => DONE_STATUS.test(String(item && item.status || ''))).length;
    // The step a person would name if asked "what is it doing?": the one in
    // progress, or the next one waiting when the agent marks nothing as such.
    const current = items.find(item => RUNNING_STATUS.test(String(item && item.status || '')))
      || items.find(item => !DONE_STATUS.test(String(item && item.status || '')));
    const title = current && current.title ? trim(current.title) : null;
    return { total: items.length, done: done, current: title };
  }

  /** The task tracker answers with `{plan: [...]}` (a fresh plan), `{tasks:
   *  [...]}` (the whole list) or `{task: {...}}` (one item's new status). A
   *  preview under the truncation cap keeps its structure, so all three are
   *  readable straight off the broadcast. */
  function readPlan(result) {
    if (!result || typeof result !== 'object') return;
    const list = Array.isArray(result.plan) ? result.plan
      : Array.isArray(result.tasks) ? result.tasks : null;
    if (list) {
      st.tasks = list.map(item => ({
        id: item && item.id, title: item && item.title, status: item && item.status,
      }));
    } else if (result.task && result.task.id) {
      // `update_task_status` reports only the item it changed.
      const updated = result.task;
      const known = st.tasks.find(item => item.id === updated.id);
      if (known) {
        known.status = updated.status;
        known.title = updated.title || known.title;
      } else {
        st.tasks.push({ id: updated.id, title: updated.title, status: updated.status });
      }
    } else {
      return;
    }
    st.plan = summarise(st.tasks);
  }

  /** The sandbox agent's plan, in the shape its service publishes:
   *  `{revision, current, progress: {total, done}, items: [{title, status}]}`.
   *  `progress` is trusted when it is there, `items` is the fallback. */
  function readSandboxPlan(plan) {
    if (!plan || typeof plan !== 'object') return null;
    const fromItems = summarise(plan.items);
    const progress = (plan.progress && typeof plan.progress === 'object') ? plan.progress : {};
    const total = Number(progress.total);
    const done = Number(progress.done);
    const current = typeof plan.current === 'string' && plan.current.trim()
      ? trim(plan.current)
      : (fromItems && fromItems.current);
    if (!Number.isFinite(total) || total <= 0) return fromItems;
    return {
      total: total,
      done: Number.isFinite(done) ? done : 0,
      current: current || null,
    };
  }

  // ── The reducer ───────────────────────────────────────────────────────────
  function feed(msg, quiet) {
    if (!msg || !msg.type) return;
    const now = Date.now();

    switch (msg.type) {
      case 'user_message':
        startRun(stamp(msg.timestamp));
        break;

      case '__typing__':
        // `showTyping()` from the legacy call sites: a run is live, but no
        // event of its own has arrived yet (e.g. right after a reconnect).
        if (st.phase === 'idle') startRun(now);
        break;

      case 'status':
        if (msg.status === 'processing') {
          if (st.phase === 'idle') startRun(now);
        } else if (st.phase !== 'idle' && st.hideAt === 0) {
          endRun('done', DONE_LINGER_MS);
        }
        break;

      case 'agent_event':
        st.lastEventAt = now;
        // The export replays the user's own turns as authored events too.
        if (msg.author && msg.author !== 'user') st.agent = msg.author;
        st.note = null;
        settle();
        break;

      case 'agent_output':
        st.lastEventAt = now;
        if (msg.agent) st.agent = msg.agent;
        pushStep(null, 'task_alt', agentLabel(msg.agent) + ' — ' + pick(TEXT.ready)).status = 'done';
        settle();
        break;

      case 'tool_activity':
        onToolActivity(msg, now);
        break;

      case 'sandbox_plan':
        // The agent inside the container rewrote its task list. This is the
        // only news that arrives during a sandbox run, which is the longest
        // single thing the system does.
        st.lastEventAt = now;
        st.sandboxPlan = readSandboxPlan(msg.plan);
        if (msg.agent) st.agent = msg.agent;
        break;

      case 'hitl_request':
        st.lastEventAt = now;
        setPhase('waiting', { agent: msg.agent_name || st.agent }, true);
        break;

      case 'hitl_timeout':
      case 'hitl_cancelled':
      case 'hitl_response':
        if (st.phase === 'waiting') { st.phase = 'thinking'; st.phraseSince = now; settle(); render(); }
        break;

      case 'final_response':
        endRun('done', DONE_LINGER_MS);
        break;

      case 'error':
        // Deliberately not showing `msg.message`: it is a Python traceback
        // summary, and it is already posted into the chat as a system message.
        st.note = null;
        endRun('error', ERROR_LINGER_MS);
        break;

      default:
        return;
    }
    if (!quiet) render();
  }

  function onToolActivity(msg, now) {
    const tool = msg.tool;
    if (!tool) return;
    st.lastEventAt = now;
    const author = msg.author || 'system';

    if (msg.phase === 'call') {
      const target = delegationTarget(tool, msg.args);
      if (target) {
        st.agent = author;
        pushStep(msg.call_id, 'alt_route', pick(PHASES.delegating) + ': ' + agentLabel(target)).status = 'done';
        setPhase('delegating', { agent: author, target: target });
        return;
      }
      const category = classify(tool, msg.description);
      const detail = detailOf(msg.args);
      if (msg.call_id) {
        st.inflight.set(msg.call_id, { tool: tool, agent: author, category: category, at: now });
      } else {
        st.anonymous++;
      }
      pushStep(msg.call_id, CATEGORIES[category].icon, phraseFor(category, tool));
      cancelSettle();
      st.note = null;
      setPhase('working', {
        category: category, detail: detail, agent: author, toolName: tool,
      });
      return;
    }

    // result / error
    const failed = msg.phase === 'error';
    if (msg.call_id && st.inflight.delete(msg.call_id)) {
      // paired with its own call
    } else if (st.anonymous > 0) {
      st.anonymous--;
    }
    closeStep(msg.call_id, failed);
    // The sandbox plan describes a container that is now done with this task.
    if (/run_sandbox_task|check_sandbox_task/.test(String(tool))) st.sandboxPlan = null;
    // The query/path shown under the phrase belonged to the call that just
    // closed — keeping it would caption the next minute of work with it.
    if (st.inflight.size + st.anonymous === 0) st.detail = null;
    if (failed) {
      // One bad tool is routine — the run continues. Say so calmly, and never
      // flip the whole indicator into an error state for it.
      st.note = pick(TEXT.toolFailed);
    } else if (/plan|task/i.test(String(tool))) {
      readPlan(msg.result);
    }
    settle();
  }

  function phraseFor(category, tool) {
    const entry = CATEGORIES[category] || CATEGORIES.tool;
    // Last resort — an unrecognised (usually freshly built MCP) tool. Its name
    // is the only honest thing left to say, so at least make it readable.
    if (category === 'tool' && tool) {
      return pick(entry) + ' «' + String(tool).replace(/_/g, ' ') + '»';
    }
    return pick(entry);
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  let renderTimer = null;
  let lastRenderAt = 0;

  function render() {
    if (!root) return;
    const now = Date.now();
    if (now - lastRenderAt < RENDER_THROTTLE_MS) {
      if (!renderTimer) {
        renderTimer = setTimeout(() => { renderTimer = null; render(); },
          RENDER_THROTTLE_MS - (now - lastRenderAt));
      }
      return;
    }
    lastRenderAt = now;
    paint();
  }

  function view() {
    // What the user should be told right now, derived fresh on every paint so
    // silence and elapsed time can change the phrase with no new events.
    let phase = st.phase;
    if (phase !== 'idle' && !connected) phase = 'offline';
    const silent = st.lastEventAt && (Date.now() - st.lastEventAt) > SILENCE_MS;
    if (silent && (phase === 'working' || phase === 'thinking' || phase === 'starting')) {
      phase = 'long_thinking';
    }

    let icon, text;
    if (phase === 'working') {
      const category = CATEGORIES[st.category] || CATEGORIES.tool;
      icon = category.icon;
      text = phraseFor(st.category, st.toolName);
    } else if (phase === 'delegating') {
      icon = PHASES.delegating.icon;
      text = pick(PHASES.delegating) + ': ' + agentLabel(st.target);
    } else {
      const entry = PHASES[phase] || PHASES.thinking;
      icon = entry.icon;
      text = pick(entry);
    }
    return { phase: phase, icon: icon, text: text };
  }

  function subLine(phase) {
    const parts = [];
    if (st.note) {
      parts.push(st.note);
    } else {
      if (st.agent && phase !== 'done' && phase !== 'stopped') parts.push(agentLabel(st.agent));
      const step = stepLine();
      // The step names the work better than a tool argument does, so it wins
      // the one slot they would otherwise share.
      if (!step && st.detail && phase === 'working') parts.push('«' + st.detail + '»');
      if (step) parts.push(step);
      if (Date.now() - st.since > LONG_RUN_MS && (phase === 'long_thinking' || phase === 'working')) {
        parts.push(pick(TEXT.longRun));
      }
    }
    return parts.join(' · ');
  }

  /** "Шаг 2 из 5: Обучить модель" — the plan position, named.
   *
   *  The sandbox agent's plan wins when there is one: it describes the step
   *  running inside the container right now, while the task tracker describes
   *  the whole study. Its counter can go *down* — an agent may add items
   *  mid-run — which is why this is a caption and never a progress bar. */
  function stepLine() {
    const plan = st.sandboxPlan || st.plan;
    if (!plan || !plan.total) return null;
    const position = Math.min(plan.done + 1, plan.total);
    const counter = pick(TEXT.step).replace('%d', position).replace('%d', plan.total);
    return plan.current ? counter + ': ' + plan.current : counter;
  }

  function paint() {
    // Nothing to paint on yet. The app's own bootstrap runs before
    // DOMContentLoaded, so `reset()` / `feed()` legitimately arrive before
    // `mount()` — the state is kept, and the first paint after mounting shows
    // it. `render()` guards the same way.
    if (!root) return;
    const current = view();
    if (current.phase === 'idle') {
      root.classList.add('hidden');
      root.innerHTML = '';
      return;
    }
    root.classList.remove('hidden');

    const tone = TONES[PHASE_TONE[current.phase] || 'work'];
    const live = ['working', 'thinking', 'starting', 'delegating', 'finalizing', 'long_thinking'].includes(current.phase);
    const sub = subLine(current.phase);
    const timer = st.since ? elapsed(Date.now() - st.since) : '';
    const steps = st.steps.slice().reverse();

    root.innerHTML = `
      <div class="si-card flex items-stretch gap-0 rounded-xl border ${tone.card} overflow-hidden transition-colors">
        <span class="w-1 shrink-0 ${tone.bar} ${live ? 'si-bar' : ''}"></span>
        <div class="flex-1 min-w-0 flex items-center gap-3 px-3 py-2.5">
          <span class="material-symbols-outlined text-lg ${tone.icon} ${live ? 'si-icon' : ''}">${current.icon}</span>
          <div class="flex-1 min-w-0">
            <div class="text-[13px] font-semibold leading-tight ${tone.text} ${live ? 'si-shimmer' : ''} truncate">${esc(current.text)}</div>
            ${sub ? `<div class="text-[10px] text-outline-variant leading-tight truncate mt-0.5">${esc(sub)}</div>` : ''}
          </div>
          ${timer ? `<span class="text-[10px] font-mono text-outline-variant shrink-0 tabular-nums">${timer}</span>` : ''}
          <button type="button" title="${esc(pick(TEXT.details))}"
            class="si-toggle shrink-0 text-outline-variant hover:text-primary transition-colors flex items-center">
            <span class="material-symbols-outlined text-base">${expanded ? 'expand_more' : 'expand_less'}</span>
          </button>
        </div>
      </div>
      ${expanded && steps.length ? `
      <div class="mt-1.5 px-3 py-2 rounded-lg border border-outline-variant/15 bg-surface-container-lowest/60 space-y-1">
        ${steps.map(step => {
          const stepTone = step.status === 'error' ? 'text-error'
            : step.status === 'done' ? 'text-outline-variant' : 'text-primary';
          const stepIcon = step.status === 'error' ? 'close'
            : step.status === 'done' ? 'check' : step.icon;
          return `<div class="flex items-center gap-2 ${stepTone}">
            <span class="material-symbols-outlined text-[13px] ${step.status === 'running' ? 'si-icon' : ''}">${stepIcon}</span>
            <span class="text-[10px] truncate">${esc(step.text)}</span>
          </div>`;
        }).join('')}
      </div>` : ''}`;

    const toggle = root.querySelector('.si-toggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        expanded = !expanded;
        try { localStorage.setItem(EXPAND_KEY, expanded ? 'on' : 'off'); } catch (_) { /* private mode */ }
        paint();
      });
    }
  }

  // The phrase can go stale on its own (silence, elapsed time), so repaint on a
  // slow tick as well as on events.
  setInterval(() => { if (st.phase !== 'idle') paint(); }, 1000);

  // ── Demo mode ─────────────────────────────────────────────────────────────
  // Replays the `agent_events.json` of a saved bundle straight into the
  // reducer, so the indicator can be developed without spending a real run.
  async function demo(filename, speed) {
    const factor = Number(speed) > 0 ? Number(speed) : 20;
    const response = await fetch('/api/saved-sessions/' + encodeURIComponent(filename) + '/events');
    if (!response.ok) throw new Error('demo: HTTP ' + response.status);
    const events = (await response.json()).events || [];
    reset();
    feed({ type: 'status', status: 'processing' });
    let previous = null;
    for (const event of events) {
      const at = stamp(event.timestamp);
      const gap = previous ? Math.min(Math.max((at - previous) / factor, 60), 2500) : 250;
      previous = at;
      await new Promise(resolve => setTimeout(resolve, gap));
      feed(event);
    }
    feed({ type: 'final_response' });
  }

  // ── Public API ────────────────────────────────────────────────────────────
  function mount(element) {
    root = element || null;
    try { expanded = localStorage.getItem(EXPAND_KEY) === 'on'; } catch (_) { expanded = false; }
    paint();
    const demoFile = new URLSearchParams(location.search).get('demo');
    if (demoFile) {
      const speed = new URLSearchParams(location.search).get('demo_speed');
      demo(demoFile, speed).catch(err => console.warn('status demo failed:', err));
    }
  }

  function reset() {
    if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
    cancelSettle();
    Object.assign(st, newState());
    paint();
  }

  /** Every public entry point is guarded.
   *
   *  This module is an observer of someone else's event stream, called from
   *  the middle of the app's own control flow — the websocket switch, the
   *  session bootstrap, the language toggle. A fault in a status line must
   *  cost the status line and nothing else; without this, one bad frame
   *  propagates into `bootstrap()`'s catch and the user is asked to create an
   *  account by hand. Same rule the sandbox and tool-activity sinks follow. */
  function guarded(name, fn) {
    return function () {
      try {
        return fn.apply(null, arguments);
      } catch (error) {
        console.warn('StatusIndicator.' + name + ' failed:', error);
        return undefined;
      }
    };
  }

  window.StatusIndicator = {
    mount: guarded('mount', mount),
    feed: guarded('feed', feed),
    reset: guarded('reset', reset),
    demo: demo,   // async: the caller already handles its rejection
    setLang: guarded('setLang', function (value) { if (value) { lang = value; paint(); } }),
    setConnected: guarded('setConnected', function (value) { connected = !!value; render(); }),
    markStopped: guarded('markStopped', function () {
      if (st.phase !== 'idle') endRun('stopped', STOPPED_LINGER_MS);
    }),
  };
})();
