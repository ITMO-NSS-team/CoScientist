// =========================================================================
// Navigation & Activity Rail
// =========================================================================
    const AGENTS = [
      { name: "OrchestratorAgent", icon: "hub", desc: "Master Orchestrator" },
      { name: "PlannerAgent", icon: "map", desc: "Roadmap Planner" },
      { name: "ToolsViewer", icon: "science", desc: "Tools Viewer" },
      { name: "KnowledgeGraph", icon: "bubble_chart", desc: "Knowledge Graph", id: "graph-link", href: "/graph" },
      { name: "MCPBuilds", icon: "build", desc: "MCP Builds", href: "/builds" },
      { name: "CoderSandbox", icon: "terminal", desc: "CoderSandbox", id: "coder-sandbox-link", href: "http://localhost:8884/" },
      { name: "__settings__", icon: "settings", desc: "Settings" },
    ];

    function isAgentHighlightedByDefault(name) {
      return true;
    }

    function initAgentNav() {
      const nav = document.getElementById('agent-nav');
      nav.innerHTML = AGENTS.map(a => {
        const highlighted = isAgentHighlightedByDefault(a.name);
        const opacityClass = highlighted ? "opacity-100" : "opacity-50";
        const iconColorClass = highlighted ? "text-primary" : "text-outline-variant";
        const textColorClass = highlighted ? "text-on-surface font-semibold" : "text-outline-variant font-medium";
        const elemIdAttr = (a.name === "KnowledgeGraph") ? 'id="graph-link"' : ((a.name === "CoderSandbox") ? 'id="coder-sandbox-link"' : `id="agent-${a.name}"`);
        const hrefAttr = a.href ? `href="${a.href}"` : '';
        const extraDot = (a.name === "CoderSandbox")
          ? `<span id="sandbox-status-dot" class="w-2 h-2 rounded-full bg-outline-variant/40 ml-auto" title="Sandbox standby"></span>`
          : '';

        return `
          <div ${elemIdAttr} ${hrefAttr} onclick="onAgentClick('${a.name}')" class="flex items-center gap-3 py-3 px-4 transition-all duration-200 ${opacityClass} cursor-pointer hover:bg-surface-variant/20 hover:opacity-100">
            <span class="material-symbols-outlined ${iconColorClass} text-lg">${a.icon}</span>
            <div class="flex flex-col">
              <span class="text-sm ${textColorClass}" data-i18n="agent.${a.name}.desc">${a.desc}</span>
              <span class="text-[8px] text-outline-variant/50 font-mono uppercase">${a.name}</span>
            </div>
            ${extraDot}
          </div>
        `;
      }).join('');
      applyLanguage();
    }

    function onAgentClick(name) {
      if (name === "__settings__") {
        openSettings();
      } else if (name === "PlannerAgent") {
        openRoadmapEditor();
      } else if (name === "ToolsViewer") {
        openToolsViewer();
      } else if (name === "KnowledgeGraph") {
        const link = document.getElementById('graph-link');
        if (link && link.href) {
          window.open(link.href, '_blank');
        } else if (activeUser && activeSession) {
          window.open(`/graph?user_id=${encodeURIComponent(activeUser.id)}&session_id=${encodeURIComponent(activeSession.id)}`, '_blank');
        } else {
          window.open('/graph', '_blank');
        }
      } else if (name === "MCPBuilds") {
        window.open('/builds', '_blank');
      } else if (name === "CoderSandbox") {
        const link = document.getElementById('coder-sandbox-link');
        const url = (link && link.href) ? link.href : (activeSandboxWatchUrl || getBaseSandboxUrl());
        window.open(url, '_blank');
      }
    }

    function highlightAgent(name) {
      AGENTS.forEach(a => {
        const targetId = a.id || ('agent-' + a.name);
        const el = document.getElementById(targetId);
        if (!el) return;
        if (a.name === name) {
          el.className = "flex items-center gap-3 py-3 px-4 bg-[#272a31] rounded-lg transition-all duration-200 border-l-2 border-[#00daf3] cursor-pointer";
          el.querySelector('.material-symbols-outlined').className = "material-symbols-outlined text-primary text-lg animate-pulse";
          el.querySelectorAll('span:not(.material-symbols-outlined)').forEach(s => s.classList.remove('text-outline-variant', 'opacity-50'));
        } else {
          const highlighted = isAgentHighlightedByDefault(a.name);
          const opacityClass = highlighted ? "opacity-100" : "opacity-50";
          el.className = `flex items-center gap-3 py-3 px-4 transition-all duration-200 ${opacityClass} cursor-pointer hover:bg-surface-variant/20 hover:opacity-100`;

          const icon = el.querySelector('.material-symbols-outlined');
          icon.className = `material-symbols-outlined ${highlighted ? 'text-primary' : 'text-outline-variant'} text-lg`;

          el.querySelectorAll('span:not(.material-symbols-outlined)').forEach(s => {
            if (!s.classList.contains('font-mono')) {
              s.className = `text-sm ${highlighted ? 'text-on-surface font-semibold' : 'text-outline-variant font-medium'}`;
            }
          });
        }
      });
    }

    function resetAgents() {
      AGENTS.forEach(a => {
        const targetId = a.id || ('agent-' + a.name);
        const el = document.getElementById(targetId);
        if (el) {
          const highlighted = isAgentHighlightedByDefault(a.name);
          const opacityClass = highlighted ? "opacity-100" : "opacity-50";
          el.className = `flex items-center gap-3 py-3 px-4 transition-all duration-200 ${opacityClass} cursor-pointer hover:bg-surface-variant/20 hover:opacity-100`;

          const icon = el.querySelector('.material-symbols-outlined');
          icon.className = `material-symbols-outlined ${highlighted ? 'text-primary' : 'text-outline-variant'} text-lg`;

          el.querySelectorAll('span:not(.material-symbols-outlined)').forEach(s => {
            if (!s.classList.contains('font-mono')) {
              s.className = `text-sm ${highlighted ? 'text-on-surface font-semibold' : 'text-outline-variant font-medium'}`;
            }
          });
        }
      });
    }


    // =========================================================================
    // Activity Rail — live view of which agents run and which tools they call
    //
    // Fed by the same ``agent_event`` stream the chat uses, but rendered as a
    // fixed two-row strip: nothing is ever appended, so a long run cannot push
    // the conversation off screen. Agent chips light up while an agent is
    // producing events; tool pills show that agent's calls with a live counter.
    // =========================================================================
    const ACTIVITY_RAIL_KEY = 'coscientist.activity_rail';
    const ACTIVITY_IDLE_MS = 20000;  // an agent with no events for this long dims

    let activityAgents = new Map();   // agent name -> { icon, tools: Map, calls, lastSeen, transferred }
    let activitySelected = null;      // agent whose tools the second row shows
    let activityPinned = false;       // user clicked a chip -> stop auto-following
    let activityTicker = null;

    const AGENT_ICONS = {
      OrchestratorAgent: 'hub',
      InitAgent: 'flag',
      PlannerAgent: 'map',
      HypothesesAgent: 'lightbulb',
      ResearchAgent: 'travel_explore',
      TaskExecutorAgent: 'alt_route',
      ToolPipelineAgent: 'checklist',
      ToolPreparerAgent: 'precision_manufacturing',
      ParallelToolSearcherAgent: 'manage_search',
      LocalToolsExtractorAgent: 'inventory_2',
      ToolRetrieverAgent: 'search',
      ToolWebSearcherAgent: 'public',
      ToolReranker: 'swap_vert',
      FullSetToolReranker: 'sort',
      WebToolsDeployerAgent: 'cloud_upload',
      McpBuilderAgent: 'construction',
      CoderAgent: 'terminal',
      DatasetCollectorAgent: 'dataset',
      MedicalAgent: 'ecg_heart',
      ExperimentAgent: 'science',
      FedotAgent: 'network_intelligence',
      ContextInitAgent: 'assignment',
      ContextInitSessionAgent: 'assignment',
      ResultAggregatorAgent: 'summarize',
    };

    const KNOWN_AGENTS = new Set(Object.keys(AGENT_ICONS));

    function agentIcon(name) {
      if (AGENT_ICONS[name]) return AGENT_ICONS[name];
      if (/tool/i.test(name)) return 'handyman';
      if (/critic|review/i.test(name)) return 'rate_review';
      if (name === 'system' || name === 'user') return 'settings_ethernet';
      return 'smart_toy';
    }

    function toolIcon(name) {
      const n = String(name || '').toLowerCase();
      if (n.includes('transfer')) return 'alt_route';
      if (n.includes('sandbox') || n.includes('shell') || n.includes('exec') || n.includes('code')) return 'terminal';
      if (n.includes('arxiv') || n.includes('pubmed') || n.includes('openalex') || n.includes('paper')) return 'menu_book';
      if (n.includes('search') || n.includes('web') || n.includes('query')) return 'travel_explore';
      if (n.includes('graph')) return 'bubble_chart';
      if (n.includes('dataset') || n.includes('data')) return 'dataset';
      if (n.includes('file') || n.includes('read') || n.includes('write') || n.includes('doc')) return 'description';
      if (n.includes('plot') || n.includes('chart') || n.includes('metric')) return 'insert_chart';
      if (n.includes('hitl') || n.includes('ask') || n.includes('input')) return 'forum';
      if (n.includes('task') || n.includes('track')) return 'checklist';
      return 'build';
    }

    function activityRailEnabled() {
      return localStorage.getItem(ACTIVITY_RAIL_KEY) !== 'off';
    }

    function toggleActivityRail() {
      localStorage.setItem(ACTIVITY_RAIL_KEY, activityRailEnabled() ? 'off' : 'on');
      renderActivityRail();
    }

    function activityAgent(name) {
      let entry = activityAgents.get(name);
      if (!entry) {
        entry = {
          name: name, icon: agentIcon(name), tools: new Map(), calls: 0, lastSeen: 0,
          running: 0, transferred: false, pending: new Set(),
        };
        activityAgents.set(name, entry);
      }
      return entry;
    }

    function activityBusy(entry) {
      return entry.pending.size + entry.running;
    }

    function activityCloseAgent(name) {
      const entry = activityAgents.get(name);
      if (!entry) return;
      entry.running = 0;
      entry.pending.clear();
      entry.transferred = false;
    }

    function activityTouchAgent(name, timestamp = null) {
      if (!name) return;
      const entry = activityAgent(name);
      entry.lastSeen = timestamp ? new Date(timestamp).getTime() : Date.now();
      if (!activityPinned) activitySelected = name;
      renderActivityRail();
    }

    function activityTool(entry, name) {
      let tool = entry.tools.get(name);
      if (!tool) {
        tool = { name: name, icon: toolIcon(name), calls: 0, done: 0, errors: 0, lastArgs: null };
        entry.tools.set(name, tool);
      }
      return tool;
    }

    function activityRecordCall(author, tc, timestamp = null) {
      const name = tc && tc.name;
      if (!name) return;
      const entry = activityAgent(author);
      entry.lastSeen = timestamp ? new Date(timestamp).getTime() : Date.now();

      // Delegation, not tool use — show the target agent as soon as it is
      // picked, before it emits anything of its own.
      const transferred = tc.target_agent || (tc.is_delegation ? tc.name : null) || (name === 'transfer_to_agent'
        ? (tc.args && (tc.args.agent_name || tc.args.agentName))
        : (KNOWN_AGENTS.has(name) || /Agent$/.test(name) ? name : null));
      if (transferred) {
        const next = activityAgent(String(transferred));
        next.transferred = true;
        next.lastSeen = entry.lastSeen;
        if (!activityPinned) activitySelected = next.name;
        renderActivityRail();
        return;
      }

      const tool = activityTool(entry, name);
      tool.calls++;
      tool.lastArgs = tc.args || null;
      entry.calls++;
      if (tc.callId) {
        entry.pending.add(tc.callId);
      } else {
        entry.running++;
      }
      if (!activityPinned) activitySelected = entry.name;
      renderActivityRail();
    }

    function activityRecordResponse(author, tr, timestamp = null) {
      const name = tr && tr.name;
      if (!name || name === 'transfer_to_agent') return;
      const isDelegation = tr.is_delegation || KNOWN_AGENTS.has(name) || /Agent$/.test(name);
      if (isDelegation) {
        activityCloseAgent(name);
        renderActivityRail();
        return;
      }
      const entry = activityAgent(author);
      entry.lastSeen = timestamp ? new Date(timestamp).getTime() : Date.now();
      const tool = activityTool(entry, name);
      tool.done++;
      if (activityResponseFailed(tr.response)) tool.errors++;
      if (tr.callId && entry.pending.delete(tr.callId)) {
        // paired with its own call
      } else {
        entry.running = Math.max(0, entry.running - 1);
      }
      renderActivityRail();
    }

    // Single entry point for the `tool_activity` stream, which reports tool use
    // at every nesting level (top-level agents and AgentTool sub-agents alike).
    function applyToolActivity(data, quiet = false) {
      const author = data.author || 'system';

      if (data.phase === 'agent_start' || data.phase === 'agent_end') {
        activityTouchAgent(author, data.timestamp);
        if (data.phase === 'agent_end') {
          activityCloseAgent(author);
        }
        if (typeof addExperimentAgentEvent === 'function') {
          addExperimentAgentEvent(author, data);
        }
        renderActivityRail();
        return;
      }

      const tool = data.tool;
      if (!tool) return;

      if (data.phase === 'call') {
        activityRecordCall(author, {
          name: tool, args: data.args, callId: data.call_id,
          is_delegation: data.is_delegation, target_agent: data.target_agent,
        }, data.timestamp);
        addExperimentToolCall(author, {
          name: tool, args: data.args, callId: data.call_id,
          truncated: !!data.args_truncated, timestamp: data.timestamp,
          parent: data.parent, is_delegation: data.is_delegation,
          target_agent: data.target_agent,
        });
        if (!quiet) addTelemetry('TOOL_CALL :: ' + author + ' → ' + tool);
        return;
      }

      const failed = data.phase === 'error';
      const response = failed ? { error: data.error } : data.result;
      const truncated = failed ? !!data.error_truncated : !!data.result_truncated;
      activityRecordResponse(author, {
        name: tool, response: response, callId: data.call_id,
        is_delegation: data.is_delegation, target_agent: data.target_agent,
      }, data.timestamp);
      addExperimentToolResponse(author, {
        name: tool, response: response, callId: data.call_id,
        truncated: truncated, failed: failed, timestamp: data.timestamp,
      });
      if (!quiet) {
        addTelemetry((failed ? 'TOOL_ERROR :: ' : 'TOOL_RESULT :: ') + author
          + (failed ? ' ✖ ' : ' ← ') + tool);
      }
    }

    function activityResponseFailed(response) {
      if (!response) return false;
      if (typeof response === 'string') return /^\s*(error|traceback)/i.test(response);
      if (typeof response !== 'object') return false;
      if (response.error) return true;
      const status = String(response.status || response.result_status || '').toLowerCase();
      return status === 'error' || status === 'failed';
    }

    function activitySelectAgent(name) {
      // Clicking the already-selected chip releases the pin and resumes
      // auto-following whichever agent is currently talking.
      if (activityPinned && activitySelected === name) {
        activityPinned = false;
      } else {
        activitySelected = name;
        activityPinned = true;
      }
      renderActivityRail();
    }

    function activityReset() {
      activityAgents = new Map();
      activitySelected = null;
      activityPinned = false;
      renderActivityRail();
    }

    function activityMarkIdle() {
      activityAgents.forEach(entry => activityCloseAgent(entry.name));
      renderActivityRail();
    }

    function renderActivityRail() {
      const rail = document.getElementById('activity-rail');
      const toggle = document.getElementById('activity-toggle');
      if (!rail) return;

      const enabled = activityRailEnabled();
      if (toggle) {
        toggle.className = enabled
          ? 'text-primary hover:brightness-125 transition-colors'
          : 'text-[#424656] hover:text-primary transition-colors';
      }
      rail.classList.toggle('hidden', !enabled || activityAgents.size === 0);
      if (!enabled || activityAgents.size === 0) return;

      const now = Date.now();
      const agents = [...activityAgents.values()].sort((a, b) => a.lastSeen - b.lastSeen);
      if (activitySelected && !activityAgents.has(activitySelected)) activitySelected = null;
      if (!activitySelected && agents.length) activitySelected = agents[agents.length - 1].name;

      document.getElementById('activity-agents').innerHTML = agents.map(entry => {
        // Silence alone means "done": an agent that has produced nothing for a
        // while is not shown as working even if a call went unclosed.
        const fresh = (now - entry.lastSeen) < ACTIVITY_IDLE_MS;
        const busy = fresh && activityBusy(entry) > 0;
        const selected = entry.name === activitySelected;
        const tone = busy
          ? 'border-primary/50 bg-primary/10 text-primary'
          : fresh
            ? 'border-outline-variant/25 bg-surface-container-high/60 text-on-surface'
            : 'border-outline-variant/10 bg-transparent text-outline-variant opacity-60';
        const ring = selected ? ' ring-1 ring-primary/40' : '';
        const pulse = busy ? ' animate-pulse' : '';
        const badge = entry.calls
          ? `<span class="text-[8px] font-mono bg-surface-container-highest/80 px-1 rounded">${entry.calls}</span>`
          : '';
        const hint = `${entry.name}${entry.calls ? ` — ${entry.calls} tool call(s)` : ''}${entry.transferred ? ' — delegated' : ''}`;
        return `
          <button onclick="activitySelectAgent('${escJs(entry.name)}')" title="${escHtml(hint)}"
            class="flex items-center gap-1.5 shrink-0 border rounded-full pl-2 pr-2.5 py-1 transition-all ${tone}${ring}">
            <span class="material-symbols-outlined text-sm${pulse}">${entry.icon}</span>
            <span class="text-[10px] font-semibold tracking-tight">${escHtml(entry.name.replace(/Agent$/, ''))}</span>
            ${badge}
          </button>`;
      }).join('');

      const selected = activityAgents.get(activitySelected);
      const tools = selected ? [...selected.tools.values()] : [];
      const toolsBox = document.getElementById('activity-tools');
      if (!tools.length) {
        toolsBox.innerHTML = `<span class="text-[9px] font-mono text-outline-variant/60">${selected ? escHtml(selected.name) + ' — no tool calls yet' : 'no tool calls yet'
          }</span>`;
        return;
      }

      const selectedFresh = (now - selected.lastSeen) < ACTIVITY_IDLE_MS;
      toolsBox.innerHTML = tools.map(tool => {
        const running = selectedFresh && tool.calls > tool.done;
        const tone = tool.errors
          ? 'border-error/40 bg-error/10 text-error'
          : running
            ? 'border-primary/40 bg-primary/10 text-primary'
            : 'border-secondary/30 bg-secondary/5 text-secondary';
        const pulse = running ? ' animate-pulse' : '';
        const count = tool.calls > 1 ? `<span class="text-[8px] font-mono opacity-80">×${tool.calls}</span>` : '';
        let hint = `${tool.name} — ${tool.done}/${tool.calls} finished`;
        if (tool.errors) hint += `, ${tool.errors} error(s)`;
        if (tool.lastArgs && Object.keys(tool.lastArgs).length) {
          hint += `\n${truncateStr(tool.lastArgs, 200)}`;
        }
        return `
          <button onclick="openToolsViewer()" title="${escHtml(hint)}"
            class="flex items-center gap-1 shrink-0 border rounded-md px-2 py-0.5 transition-all ${tone}">
            <span class="material-symbols-outlined text-[13px]${pulse}">${tool.icon}</span>
            <span class="text-[9px] font-mono tracking-tight">${escHtml(tool.name)}</span>
            ${count}
          </button>`;
      }).join('');
    }

    // Re-render on a slow tick so chips fade to "idle" without new events.
    activityTicker = setInterval(() => {
      if (activityAgents.size) renderActivityRail();
    }, 5000);


    // =========================================================================
    // Layout — collapsible left rail
    // =========================================================================
    function applySideNavState() {
      const collapsed = localStorage.getItem(SIDE_NAV_KEY) === 'off';
      document.body.classList.toggle('nav-collapsed', collapsed);
      const icon = document.getElementById('side-nav-toggle-icon');
      icon.textContent = collapsed ? 'left_panel_open' : 'left_panel_close';
      document.getElementById('side-nav-toggle').title = collapsed ? 'Show sidebar' : 'Hide sidebar';
    }

    function toggleSideNav() {
      const collapsed = document.body.classList.contains('nav-collapsed');
      localStorage.setItem(SIDE_NAV_KEY, collapsed ? 'on' : 'off');
      applySideNavState();
    }

