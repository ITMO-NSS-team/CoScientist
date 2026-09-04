// =========================================================================
// Roadmap & Execution Plan Modal (Visual & YAML Views)
// =========================================================================
(function () {
  'use strict';

  let currentTasks = [];
  let currentView = 'visual'; // 'visual' | 'yaml'
  let activeFilter = 'all';    // 'all' | 'active' | 'done' | 'todo'
  let searchQuery = '';
  let isSaving = false;

  const AGENT_ICONS = {
    OrchestratorAgent: 'hub',
    PlannerAgent: 'map',
    PlanningPipelineAgent: 'map',
    HypothesesAgent: 'lightbulb',
    ResearchAgent: 'travel_explore',
    TaskExecutorAgent: 'alt_route',
    CoderAgent: 'terminal',
    DatasetCollectorAgent: 'dataset',
    MedicalAgent: 'medical_services',
    McpBuilderAgent: 'construction',
    ToolPipelineAgent: 'checklist',
    ToolPreparerAgent: 'precision_manufacturing',
    ResultAggregatorAgent: 'summarize',
    FedotAgent: 'auto_graph',
    system: 'settings_suggest',
  };

  const STATUS_CONFIG = {
    done: {
      label: 'Completed',
      icon: 'check_circle',
      badgeClass: 'text-secondary bg-secondary/10 border-secondary/30',
      barClass: 'bg-secondary',
      spin: false,
    },
    in_progress: {
      label: 'In Progress',
      icon: 'autorenew',
      badgeClass: 'text-primary bg-primary/10 border-primary/40 animate-pulse',
      barClass: 'bg-primary shadow-[0_0_8px_rgba(0,218,243,0.5)]',
      spin: true,
    },
    error: {
      label: 'Failed',
      icon: 'error',
      badgeClass: 'text-error bg-error/10 border-error/30',
      barClass: 'bg-error',
      spin: false,
    },
    todo: {
      label: 'Pending',
      icon: 'schedule',
      badgeClass: 'text-outline-variant bg-surface-container-high border-outline-variant/20',
      barClass: 'bg-outline-variant/40',
      spin: false,
    },
  };

  function normalizeStatus(status) {
    const s = String(status || '').toLowerCase().trim();
    if (/done|complete|finish|success/.test(s)) return 'done';
    if (/progress|running|active|doing/.test(s)) return 'in_progress';
    if (/fail|error|cancel/.test(s)) return 'error';
    return 'todo';
  }

  function getAgentIcon(assignee) {
    if (!assignee) return 'smart_toy';
    return AGENT_ICONS[assignee] || 'smart_toy';
  }

  function agentShortName(name) {
    if (!name) return 'Agent';
    return String(name).replace(/Agent$/, '');
  }

  function escHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── YAML Serialization & Parsing ──────────────────────────────────────────
  function tasksToYaml(tasks) {
    if (window.jsyaml && typeof window.jsyaml.dump === 'function') {
      try {
        return window.jsyaml.dump(tasks, {
          indent: 2,
          lineWidth: -1,
          noRefs: true,
          quotingType: '"',
        });
      } catch (err) {
        console.warn('js-yaml dump failed, falling back:', err);
      }
    }
    // Lightweight fallback YAML serializer
    if (!Array.isArray(tasks)) return '[]\n';
    return tasks.map(t => {
      let lines = [`- id: ${JSON.stringify(t.id || '')}`];
      lines.push(`  title: ${JSON.stringify(t.title || '')}`);
      lines.push(`  assignee: ${JSON.stringify(t.assignee || '')}`);
      lines.push(`  status: ${JSON.stringify(t.status || 'TODO')}`);
      if (t.parent_id !== undefined) lines.push(`  parent_id: ${t.parent_id ? JSON.stringify(t.parent_id) : 'null'}`);
      if (t.description) {
        const desc = String(t.description);
        if (desc.includes('\n')) {
          lines.push(`  description: |`);
          desc.split('\n').forEach(l => lines.push(`    ${l}`));
        } else {
          lines.push(`  description: ${JSON.stringify(desc)}`);
        }
      }
      if (t.notes) lines.push(`  notes: ${JSON.stringify(t.notes)}`);
      return lines.join('\n');
    }).join('\n\n') + '\n';
  }

  function yamlToTasks(yamlText) {
    if (!yamlText || !yamlText.trim()) return [];
    if (window.jsyaml && typeof window.jsyaml.load === 'function') {
      const parsed = window.jsyaml.load(yamlText);
      if (Array.isArray(parsed)) return parsed;
      if (parsed && typeof parsed === 'object') {
        if (Array.isArray(parsed.tasks)) return parsed.tasks;
        if (Array.isArray(parsed.plan)) return parsed.plan;
      }
      throw new Error('YAML must contain a list of tasks (- id: ...)');
    }
    // Fallback: try parsing as JSON first
    try {
      const j = JSON.parse(yamlText);
      if (Array.isArray(j)) return j;
    } catch (_) {}
    throw new Error('YAML parser unavailable or invalid format');
  }

  // ── Open / Close Modal ───────────────────────────────────────────────────
  async function openRoadmapEditor() {
    const modal = document.getElementById('roadmap-modal');
    const status = document.getElementById('roadmap-status');
    const saveBtn = document.getElementById('save-roadmap-btn');

    modal.classList.remove('hidden');
    if (status) {
      status.classList.remove('hidden');
      status.textContent = 'Loading roadmap…';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
    }
    if (saveBtn) saveBtn.disabled = true;

    updateRoadmapModalButtons();

    try {
      const response = await fetch(roadmapUrl());
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (Array.isArray(data.tasks)) {
        currentTasks = data.tasks;
      } else if (data.content && data.content.trim()) {
        try {
          const parsed = JSON.parse(data.content);
          currentTasks = Array.isArray(parsed) ? parsed : (parsed.tasks || []);
        } catch (_) {
          currentTasks = [];
        }
      } else {
        currentTasks = [];
      }

      if (status) {
        status.textContent = 'Roadmap loaded.';
        status.className = 'text-[10px] font-mono text-secondary';
        setTimeout(() => { if (status) status.classList.add('hidden'); }, 1800);
      }
    } catch (error) {
      console.error('Failed to load roadmap:', error);
      if (status) {
        status.textContent = 'Error loading roadmap: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    } finally {
      if (saveBtn) saveBtn.disabled = false;
      renderRoadmapView();
    }
  }

  function closeRoadmapEditor() {
    const modal = document.getElementById('roadmap-modal');
    if (modal) modal.classList.add('hidden');
  }

  // ── View Switching ───────────────────────────────────────────────────────
  function switchRoadmapView(view) {
    if (view === currentView) return;

    if (view === 'yaml') {
      // Switch from Visual to YAML: serialize tasks
      currentView = 'yaml';
      updateViewButtons();
      renderRoadmapYaml();
    } else {
      // Switch from YAML to Visual: parse YAML
      try {
        const textarea = document.getElementById('roadmap-textarea');
        if (textarea) {
          const parsed = yamlToTasks(textarea.value);
          currentTasks = parsed;
        }
        currentView = 'visual';
        updateViewButtons();
        renderRoadmapVisual();
      } catch (err) {
        const yamlStatus = document.getElementById('roadmap-yaml-status');
        if (yamlStatus) {
          yamlStatus.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-error"></span><span class="text-error">Syntax error: ${escHtml(err.message)}</span>`;
        }
        alert('Cannot switch to Visual Plan: please fix YAML syntax errors first:\n' + err.message);
      }
    }
  }

  function updateViewButtons() {
    const visualBtn = document.getElementById('roadmap-view-visual-btn');
    const yamlBtn = document.getElementById('roadmap-view-yaml-btn');
    const visualView = document.getElementById('roadmap-visual-view');
    const yamlView = document.getElementById('roadmap-yaml-view');
    const toolbar = document.getElementById('roadmap-visual-toolbar');

    if (currentView === 'visual') {
      if (visualBtn) {
        visualBtn.className = 'flex items-center gap-1 px-2.5 py-1 rounded-md bg-primary text-on-primary shadow transition-all font-semibold';
      }
      if (yamlBtn) {
        yamlBtn.className = 'flex items-center gap-1 px-2.5 py-1 rounded-md text-outline-variant hover:text-on-surface transition-all';
      }
      if (visualView) visualView.classList.remove('hidden');
      if (yamlView) yamlView.classList.add('hidden');
      if (toolbar) toolbar.classList.remove('hidden');
    } else {
      if (visualBtn) {
        visualBtn.className = 'flex items-center gap-1 px-2.5 py-1 rounded-md text-outline-variant hover:text-on-surface transition-all';
      }
      if (yamlBtn) {
        yamlBtn.className = 'flex items-center gap-1 px-2.5 py-1 rounded-md bg-primary text-on-primary shadow transition-all font-semibold';
      }
      if (visualView) visualView.classList.add('hidden');
      if (yamlView) yamlView.classList.remove('hidden');
      if (toolbar) toolbar.classList.add('hidden');
    }
  }

  // ── Render Master ────────────────────────────────────────────────────────
  function renderRoadmapView() {
    updateProgressAndStats();
    if (currentView === 'visual') {
      renderRoadmapVisual();
    } else {
      renderRoadmapYaml();
    }
  }

  function updateProgressAndStats() {
    const total = currentTasks.length;
    let done = 0;
    let active = 0;
    let todo = 0;

    currentTasks.forEach(t => {
      const norm = normalizeStatus(t.status);
      if (norm === 'done') done++;
      else if (norm === 'in_progress') active++;
      else todo++;
    });

    const percent = total > 0 ? Math.round((done / total) * 100) : 0;

    const pBar = document.getElementById('roadmap-progress-bar');
    const pText = document.getElementById('roadmap-progress-text');
    const cBadge = document.getElementById('roadmap-counter-badge');

    if (pBar) pBar.style.width = percent + '%';
    if (pText) pText.textContent = `${done} of ${total} completed (${percent}%)`;
    if (cBadge) cBadge.textContent = `${total} task${total === 1 ? '' : 's'}`;

    // Filter counts
    const fAll = document.getElementById('rf-all-count');
    const fActive = document.getElementById('rf-active-count');
    const fDone = document.getElementById('rf-done-count');
    const fTodo = document.getElementById('rf-todo-count');
    if (fAll) fAll.textContent = total;
    if (fActive) fActive.textContent = active;
    if (fDone) fDone.textContent = done;
    if (fTodo) fTodo.textContent = todo;

    // Footer stats
    const sTotal = document.getElementById('roadmap-stat-total');
    const sActive = document.getElementById('roadmap-stat-active');
    const sDone = document.getElementById('roadmap-stat-done');
    if (sTotal) sTotal.textContent = `Total: ${total}`;
    if (sActive) sActive.textContent = `In progress: ${active}`;
    if (sDone) sDone.textContent = `Done: ${done}`;
  }

  // ── Visual View Rendering ────────────────────────────────────────────────
  function renderRoadmapVisual() {
    const listEl = document.getElementById('roadmap-tasks-list');
    const emptyEl = document.getElementById('roadmap-empty-state');
    if (!listEl) return;

    let filtered = currentTasks.filter(t => {
      const norm = normalizeStatus(t.status);
      if (activeFilter === 'active' && norm !== 'in_progress') return false;
      if (activeFilter === 'done' && norm !== 'done') return false;
      if (activeFilter === 'todo' && (norm === 'done' || norm === 'in_progress')) return false;

      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const str = `${t.id || ''} ${t.title || ''} ${t.description || ''} ${t.assignee || ''} ${t.notes || ''}`.toLowerCase();
        if (!str.includes(q)) return false;
      }
      return true;
    });

    if (!currentTasks.length || !filtered.length) {
      listEl.innerHTML = '';
      if (emptyEl) {
        emptyEl.classList.remove('hidden');
        emptyEl.querySelector('p.text-sm').textContent = currentTasks.length === 0
          ? 'No tasks found in roadmap'
          : 'No tasks match current filter/search';
      }
      return;
    }

    if (emptyEl) emptyEl.classList.add('hidden');

    listEl.innerHTML = filtered.map((task, idx) => {
      const norm = normalizeStatus(task.status);
      const cfg = STATUS_CONFIG[norm] || STATUS_CONFIG.todo;
      const agentIcon = getAgentIcon(task.assignee);
      const agentName = agentShortName(task.assignee);
      const taskId = task.id || `TASK-${idx + 1}`;
      const desc = task.description || '';
      const isLongDesc = desc.length > 200;

      return `
        <div id="roadmap-card-${escHtml(taskId)}" class="roadmap-card flex items-stretch gap-0 rounded-xl border border-outline-variant/15 bg-surface-container-lowest/80 overflow-hidden group">
          <!-- Left Status Indicator Bar -->
          <span class="w-1.5 shrink-0 ${cfg.barClass}"></span>

          <!-- Card Content -->
          <div class="flex-1 p-4 min-w-0 flex flex-col gap-2.5">
            <!-- Header row: ID, Status, Assignee, Parent Dep -->
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="flex flex-wrap items-center gap-2">
                <!-- Task ID -->
                <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-surface-container-high text-on-surface border border-outline-variant/20">
                  ${escHtml(taskId)}
                </span>

                <!-- Status Badge (Clickable cycle) -->
                <button type="button" onclick="cycleTaskStatus('${escHtml(taskId)}')"
                  title="Click to change status"
                  class="flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded-full border transition-all hover:scale-105 ${cfg.badgeClass}">
                  <span class="material-symbols-outlined text-[13px] ${cfg.spin ? 'animate-spin' : ''}">${cfg.icon}</span>
                  <span>${cfg.label}</span>
                </button>

                <!-- Assignee Agent Chip -->
                <span class="flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full bg-surface-container-high/80 border border-outline-variant/15 text-on-surface">
                  <span class="material-symbols-outlined text-xs text-primary">${agentIcon}</span>
                  <span>${escHtml(agentName)}</span>
                </span>

                <!-- Parent Dependency -->
                ${task.parent_id ? `
                  <button type="button" onclick="highlightParentTask('${escHtml(task.parent_id)}')"
                    title="Jump to prerequisite task"
                    class="flex items-center gap-1 text-[9px] font-mono px-2 py-0.5 rounded-full bg-primary/5 text-primary border border-primary/20 hover:bg-primary/10 transition-colors">
                    <span class="material-symbols-outlined text-[11px]">arrow_upward</span>
                    <span>Depends on: ${escHtml(task.parent_id)}</span>
                  </button>
                ` : ''}
              </div>

              <!-- Action Menu: Delete / Edit -->
              <div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
                <button type="button" onclick="deleteRoadmapTask('${escHtml(taskId)}')"
                  class="p-1 rounded text-outline-variant hover:text-error hover:bg-error/10 transition-colors"
                  title="Delete task">
                  <span class="material-symbols-outlined text-sm">delete</span>
                </button>
              </div>
            </div>

            <!-- Task Title -->
            <div class="text-sm font-semibold text-on-surface leading-snug tracking-tight">
              ${escHtml(task.title || 'Untitled task')}
            </div>

            <!-- Task Description -->
            ${desc ? `
              <div class="relative">
                <div id="desc-${escHtml(taskId)}" class="text-xs text-on-surface-variant leading-relaxed font-sans roadmap-desc-clamp ${isLongDesc ? '' : 'expanded'}">
                  ${escHtml(desc)}
                </div>
                ${isLongDesc ? `
                  <button type="button" onclick="toggleDescExpand('${escHtml(taskId)}', this)"
                    class="mt-1 text-[10px] font-semibold text-primary hover:underline flex items-center gap-0.5">
                    <span>Show more</span>
                    <span class="material-symbols-outlined text-xs">expand_more</span>
                  </button>
                ` : ''}
              </div>
            ` : ''}

            <!-- Task Notes / Details -->
            ${task.notes ? `
              <div class="flex items-start gap-1.5 text-[10px] text-outline-variant/80 bg-surface-container-high/40 border border-outline-variant/10 rounded-md px-2.5 py-1.5">
                <span class="material-symbols-outlined text-xs text-primary/70 shrink-0 mt-0.5">info</span>
                <span class="italic leading-tight">${escHtml(task.notes)}</span>
              </div>
            ` : ''}
          </div>
        </div>
      `;
    }).join('');
  }

  // ── YAML View Rendering ──────────────────────────────────────────────────
  function renderRoadmapYaml() {
    const textarea = document.getElementById('roadmap-textarea');
    if (!textarea) return;
    textarea.value = tasksToYaml(currentTasks);
    validateYamlString(textarea.value);
  }

  function handleRoadmapYamlInput(value) {
    validateYamlString(value);
  }

  function validateYamlString(text) {
    const statusEl = document.getElementById('roadmap-yaml-status');
    if (!statusEl) return;
    try {
      const parsed = yamlToTasks(text);
      statusEl.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-secondary"></span><span class="text-secondary">YAML valid · ${parsed.length} tasks</span>`;
      return true;
    } catch (err) {
      statusEl.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-error"></span><span class="text-error">${escHtml(err.message)}</span>`;
      return false;
    }
  }

  function formatRoadmapYaml() {
    const textarea = document.getElementById('roadmap-textarea');
    if (!textarea) return;
    try {
      const parsed = yamlToTasks(textarea.value);
      currentTasks = parsed;
      textarea.value = tasksToYaml(parsed);
      validateYamlString(textarea.value);
      updateProgressAndStats();
    } catch (err) {
      alert('Cannot format YAML with syntax errors: ' + err.message);
    }
  }

  function copyRoadmapYaml() {
    const textarea = document.getElementById('roadmap-textarea');
    if (!textarea) return;
    navigator.clipboard.writeText(textarea.value).then(() => {
      const status = document.getElementById('roadmap-status');
      if (status) {
        status.classList.remove('hidden');
        status.textContent = 'YAML copied to clipboard.';
        status.className = 'text-[10px] font-mono text-secondary';
        setTimeout(() => status.classList.add('hidden'), 1500);
      }
    }).catch(err => console.error('Copy failed:', err));
  }

  // ── Task Actions ─────────────────────────────────────────────────────────
  function cycleTaskStatus(taskId) {
    const task = currentTasks.find(t => (t.id || '') === taskId);
    if (!task) return;
    const norm = normalizeStatus(task.status);
    const order = ['todo', 'in_progress', 'done'];
    const next = order[(order.indexOf(norm) + 1) % order.length];

    task.status = next === 'done' ? 'DONE' : next === 'in_progress' ? 'IN_PROGRESS' : 'TODO';
    updateProgressAndStats();
    renderRoadmapVisual();
    // Sync with status indicator
    if (window.StatusIndicator) {
      StatusIndicator.feed({ type: 'tasks_updated', tasks: currentTasks });
    }
  }

  function toggleDescExpand(taskId, btn) {
    const descEl = document.getElementById(`desc-${taskId}`);
    if (!descEl) return;
    const isExp = descEl.classList.toggle('expanded');
    btn.innerHTML = isExp
      ? `<span>Show less</span><span class="material-symbols-outlined text-xs">expand_less</span>`
      : `<span>Show more</span><span class="material-symbols-outlined text-xs">expand_more</span>`;
  }

  function highlightParentTask(parentId) {
    const targetCard = document.getElementById(`roadmap-card-${parentId}`);
    if (targetCard) {
      targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
      targetCard.classList.remove('highlighted');
      // Trigger reflow to restart animation
      void targetCard.offsetWidth;
      targetCard.classList.add('highlighted');
    } else {
      alert(`Prerequisite task '${parentId}' not found in the plan.`);
    }
  }

  function deleteRoadmapTask(taskId) {
    if (!confirm(`Delete task ${taskId}?`)) return;
    currentTasks = currentTasks.filter(t => (t.id || '') !== taskId);
    updateProgressAndStats();
    renderRoadmapVisual();
  }

  function filterRoadmapTasks(filter) {
    activeFilter = filter;
    document.querySelectorAll('#roadmap-filter-tabs .roadmap-tab-btn').forEach(btn => {
      const f = btn.dataset.filter;
      if (f === filter) {
        btn.className = 'roadmap-tab-btn active px-2.5 py-1 rounded-md bg-surface-container-highest text-primary font-semibold border border-primary/20 transition-all';
      } else {
        btn.className = 'roadmap-tab-btn px-2.5 py-1 rounded-md text-outline-variant hover:text-on-surface transition-all';
      }
    });
    renderRoadmapVisual();
  }

  function searchRoadmapTasks(query) {
    searchQuery = (query || '').trim();
    renderRoadmapVisual();
  }

  // ── Save & Confirm Operations ────────────────────────────────────────────
  async function prepareTasksForSave() {
    if (currentView === 'yaml') {
      const textarea = document.getElementById('roadmap-textarea');
      if (textarea) {
        currentTasks = yamlToTasks(textarea.value);
      }
    }
    return currentTasks;
  }

  async function saveRoadmap() {
    if (isSaving) return;
    const status = document.getElementById('roadmap-status');
    const saveBtn = document.getElementById('save-roadmap-btn');

    try {
      const tasks = await prepareTasksForSave();
      isSaving = true;
      if (saveBtn) saveBtn.disabled = true;

      if (status) {
        status.classList.remove('hidden');
        status.textContent = 'Saving changes…';
        status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
      }

      const response = await fetch(roadmapUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: JSON.stringify(tasks, null, 2) }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.status === 'success') {
        if (status) {
          status.textContent = 'Roadmap saved successfully.';
          status.className = 'text-[10px] font-mono text-secondary';
        }
        addTelemetry('ROADMAP :: saved successfully');

        // Instantly notify StatusIndicator
        if (window.StatusIndicator) {
          StatusIndicator.feed({ type: 'tasks_updated', tasks: tasks });
        }

        setTimeout(closeRoadmapEditor, 800);
      } else {
        throw new Error(data.error || 'Failed to save roadmap');
      }
    } catch (error) {
      console.error('Failed to save roadmap:', error);
      if (status) {
        status.textContent = 'Error saving roadmap: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    } finally {
      isSaving = false;
      if (saveBtn) saveBtn.disabled = false;
    }
  }

  function updateRoadmapModalButtons() {
    const confirmBtn = document.getElementById('confirm-roadmap-btn');
    const reviseBtn = document.getElementById('revise-roadmap-btn');
    const feedbackContainer = document.getElementById('roadmap-feedback-container');
    if (!confirmBtn) return;
    if (window.currentPlannerHitlRequest) {
      confirmBtn.classList.remove('hidden');
      if (reviseBtn) reviseBtn.classList.remove('hidden');
      if (feedbackContainer) feedbackContainer.classList.remove('hidden');
    } else {
      confirmBtn.classList.add('hidden');
      if (reviseBtn) reviseBtn.classList.add('hidden');
      if (feedbackContainer) feedbackContainer.classList.add('hidden');
    }
  }

  async function reviseRoadmap() {
    if (!window.currentPlannerHitlRequest) {
      alert('No pending roadmap confirmation request found.');
      return;
    }

    const feedbackInput = document.getElementById('roadmap-feedback-input');
    const status = document.getElementById('roadmap-status');
    const feedback = feedbackInput ? feedbackInput.value.trim() : '';

    if (status) {
      status.classList.remove('hidden');
      status.textContent = 'Sending revision request…';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
    }

    try {
      const tasks = await prepareTasksForSave();
      const response = await fetch(roadmapUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: JSON.stringify(tasks, null, 2) }),
      });

      if (response.ok) {
        const reqId = window.currentPlannerHitlRequest.request_id;
        if (window.ws && window.ws.readyState === 1) {
          window.ws.send(JSON.stringify({
            type: 'hitl_response',
            request_id: reqId,
            action: 'edit',
            approved: false,
            instructions: feedback || null,
            free_input: feedback || null,
          }));
        }
        if (window.StatusIndicator) {
          StatusIndicator.feed({ type: 'hitl_response', request_id: reqId });
          StatusIndicator.feed({ type: 'tasks_updated', tasks: tasks });
        }

        const hitlPanel = document.getElementById('hitl-panel');
        if (hitlPanel) hitlPanel.classList.add('hidden');
        addSystemMsg('✗ HITL Sent for Revision' + (feedback ? ': ' + feedback : ''));

        window.currentPlannerHitlRequest = null;
        updateRoadmapModalButtons();
        if (status) {
          status.textContent = 'Revision requested.';
          status.className = 'text-[10px] font-mono text-error';
        }
        setTimeout(closeRoadmapEditor, 900);
      }
    } catch (error) {
      console.error('Error revising roadmap:', error);
      if (status) {
        status.textContent = 'Error: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    }
  }

  async function saveAndConfirmRoadmap() {
    if (!window.currentPlannerHitlRequest) {
      alert('No pending roadmap confirmation request found.');
      return;
    }

    const status = document.getElementById('roadmap-status');
    const saveBtn = document.getElementById('save-roadmap-btn');
    const confirmBtn = document.getElementById('confirm-roadmap-btn');

    if (status) {
      status.classList.remove('hidden');
      status.textContent = 'Saving and confirming plan…';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
    }
    if (saveBtn) saveBtn.disabled = true;
    if (confirmBtn) confirmBtn.disabled = true;

    try {
      const tasks = await prepareTasksForSave();
      const response = await fetch(roadmapUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: JSON.stringify(tasks, null, 2) }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.status === 'success') {
        if (status) {
          status.textContent = 'Roadmap saved and confirmed.';
          status.className = 'text-[10px] font-mono text-secondary';
        }
        addTelemetry('ROADMAP :: saved successfully');

        const reqId = window.currentPlannerHitlRequest.request_id;
        respondHITL(reqId, true);

        if (window.StatusIndicator) {
          StatusIndicator.feed({ type: 'tasks_updated', tasks: tasks });
        }

        setTimeout(closeRoadmapEditor, 900);
      } else {
        throw new Error(data.error || 'Unknown error');
      }
    } catch (error) {
      console.error('Failed to save and confirm roadmap:', error);
      if (status) {
        status.textContent = 'Error: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    } finally {
      if (saveBtn) saveBtn.disabled = false;
      if (confirmBtn) confirmBtn.disabled = false;
    }
  }

  // ── Global Window Exports ────────────────────────────────────────────────
  window.openRoadmapEditor = openRoadmapEditor;
  window.closeRoadmapEditor = closeRoadmapEditor;
  window.saveRoadmap = saveRoadmap;
  window.reviseRoadmap = reviseRoadmap;
  window.saveAndConfirmRoadmap = saveAndConfirmRoadmap;
  window.updateRoadmapModalButtons = updateRoadmapModalButtons;
  window.switchRoadmapView = switchRoadmapView;
  window.formatRoadmapYaml = formatRoadmapYaml;
  window.copyRoadmapYaml = copyRoadmapYaml;
  window.handleRoadmapYamlInput = handleRoadmapYamlInput;
  window.cycleTaskStatus = cycleTaskStatus;
  window.toggleDescExpand = toggleDescExpand;
  window.highlightParentTask = highlightParentTask;
  window.deleteRoadmapTask = deleteRoadmapTask;
  window.filterRoadmapTasks = filterRoadmapTasks;
  window.searchRoadmapTasks = searchRoadmapTasks;
})();
