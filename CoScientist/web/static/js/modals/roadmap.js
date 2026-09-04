// =========================================================================
// Roadmap Editor Modal
// =========================================================================
    // =========================================================================
    // Roadmap Editor
    // =========================================================================
    async function openRoadmapEditor() {
      const modal = document.getElementById('roadmap-modal');
      const textarea = document.getElementById('roadmap-textarea');
      const status = document.getElementById('roadmap-status');
      const saveBtn = document.getElementById('save-roadmap-btn');

      // Show modal
      modal.classList.remove('hidden');
      status.classList.remove('hidden');
      status.textContent = 'Loading the current session roadmap...';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
      textarea.disabled = true;
      saveBtn.disabled = true;

      updateRoadmapModalButtons();

      try {
        const response = await fetch(roadmapUrl());
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        textarea.value = data.content || '';
        status.textContent = 'Roadmap loaded successfully.';
        status.className = 'text-[10px] font-mono text-secondary';
      } catch (error) {
        console.error('Failed to load roadmap:', error);
        status.textContent = 'Error loading roadmap: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      } finally {
        textarea.disabled = false;
        saveBtn.disabled = false;
      }
    }

    function closeRoadmapEditor() {
      const modal = document.getElementById('roadmap-modal');
      modal.classList.add('hidden');
    }

    async function saveRoadmap() {
      const textarea = document.getElementById('roadmap-textarea');
      const status = document.getElementById('roadmap-status');
      const saveBtn = document.getElementById('save-roadmap-btn');
      const content = textarea.value;

      status.classList.remove('hidden');
      status.textContent = 'Saving changes...';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
      textarea.disabled = true;
      saveBtn.disabled = true;

      try {
        const response = await fetch(roadmapUrl(), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ content: content })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.status === 'success') {
          status.textContent = 'Roadmap saved successfully.';
          status.className = 'text-[10px] font-mono text-secondary';
          // Auto close after 1 second
          setTimeout(closeRoadmapEditor, 1000);
          addTelemetry('ROADMAP :: saved successfully');
        } else {
          throw new Error(data.error || 'Unknown error');
        }
      } catch (error) {
        console.error('Failed to save roadmap:', error);
        status.textContent = 'Error saving roadmap: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      } finally {
        textarea.disabled = false;
        saveBtn.disabled = false;
      }
    }

    function updateRoadmapModalButtons() {
      const confirmBtn = document.getElementById('confirm-roadmap-btn');
      const reviseBtn = document.getElementById('revise-roadmap-btn');
      const feedbackContainer = document.getElementById('roadmap-feedback-container');
      if (!confirmBtn) return;
      if (currentPlannerHitlRequest) {
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
      if (!currentPlannerHitlRequest) {
        alert("No pending roadmap confirmation request found.");
        return;
      }

      const textarea = document.getElementById('roadmap-textarea');
      const feedbackInput = document.getElementById('roadmap-feedback-input');
      const status = document.getElementById('roadmap-status');
      const content = textarea.value;
      const feedback = feedbackInput ? feedbackInput.value : '';

      status.classList.remove('hidden');
      status.textContent = 'Sending revision request...';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';

      try {
        const response = await fetch(roadmapUrl(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: content })
        });

        if (response.ok) {
          if (ws && ws.readyState === 1) {
            ws.send(JSON.stringify({
              type: 'hitl_response',
              request_id: currentPlannerHitlRequest.request_id,
              action: 'edit',
              approved: false,
              instructions: feedback || null,
              free_input: feedback || null,
            }));
          }

          document.getElementById('hitl-panel').classList.add('hidden');
          addSystemMsg('✗ HITL Sent for Revision' + (feedback ? ': ' + feedback : ''));

          currentPlannerHitlRequest = null;
          updateRoadmapModalButtons();
          setTimeout(closeRoadmapEditor, 1000);
          status.textContent = 'Revision requested.';
          status.className = 'text-[10px] font-mono text-error';
        }
      } catch (error) {
        console.error('Error:', error);
      }
    }

    async function saveAndConfirmRoadmap() {
      if (!currentPlannerHitlRequest) {
        alert("No pending roadmap confirmation request found.");
        return;
      }

      const textarea = document.getElementById('roadmap-textarea');
      const status = document.getElementById('roadmap-status');
      const saveBtn = document.getElementById('save-roadmap-btn');
      const confirmBtn = document.getElementById('confirm-roadmap-btn');
      const content = textarea.value;

      status.classList.remove('hidden');
      status.textContent = 'Saving and confirming plan...';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
      textarea.disabled = true;
      saveBtn.disabled = true;
      confirmBtn.disabled = true;

      try {
        const response = await fetch(roadmapUrl(), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ content: content })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.status === 'success') {
          status.textContent = 'Roadmap saved and confirmed successfully.';
          status.className = 'text-[10px] font-mono text-secondary';
          addTelemetry('ROADMAP :: saved successfully');

          const reqId = currentPlannerHitlRequest.request_id;
          respondHITL(reqId, true);

          setTimeout(closeRoadmapEditor, 1000);
        } else {
          throw new Error(data.error || 'Unknown error');
        }
      } catch (error) {
        console.error('Failed to save and confirm roadmap:', error);
        status.textContent = 'Error: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      } finally {
        textarea.disabled = false;
        saveBtn.disabled = false;
        confirmBtn.disabled = false;
      }
    }
