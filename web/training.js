/**
 * Netcradus Training Panel — Frontend Controller.
 *
 * Features:
 * - Dataset management (upload, delete, validate).
 * - Training control (start, stop, pause, resume).
 * - Real-time progress with ETA and resource monitoring.
 * - Loss graph visualization.
 * - Checkpoint management (save, load, delete, export).
 * - Live log viewer with filtering.
 * - Dark/Light theme toggle.
 * - Toast notifications.
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // DOM references
  // ---------------------------------------------------------------------------
  const tpContent = document.getElementById("tp-content");
  const toastContainer = document.getElementById("toast-container");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const themeIcon = document.getElementById("theme-icon");
  const btnThemeToggle = document.getElementById("btn-theme-toggle");

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let currentTab = "datasets";
  let pollingInterval = null;
  let logAutoRefresh = null;

  // ---------------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------------
  function initTheme() {
    const saved = localStorage.getItem("netcradus_theme") || "dark";
    document.documentElement.setAttribute("data-theme", saved);
    updateThemeIcon(saved);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("netcradus_theme", next);
    updateThemeIcon(next);
  }

  function updateThemeIcon(theme) {
    if (!themeIcon) return;
    if (theme === "dark") {
      themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    } else {
      themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
    }
  }

  if (btnThemeToggle) {
    btnThemeToggle.addEventListener("click", toggleTheme);
  }

   // ---------------------------------------------------------------------------
   // API helper
   // ---------------------------------------------------------------------------
   async function api(path, options = {}) {
     const headers = { "Content-Type": "application/json" };
     const res = await fetch(`/api/${path}`, { headers, ...options });
     const data = await res.json().catch(() => ({}));
     if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
     return data;
   }

   function apiError(err) {
     if (err.name === "TypeError" && err.message === "Failed to fetch") {
       return "Network error — the server may be down or unreachable. Please check your connection and try again.";
     }
     if (err.message.includes("HTTP 404")) {
       return "The requested resource was not found.";
     }
     if (err.message.includes("HTTP 500")) {
       return "An internal server error occurred. Please try again later.";
     }
     if (err.message.includes("HTTP 403")) {
       return "Access denied. You do not have permission to perform this action.";
     }
     if (err.message.includes("HTTP 400")) {
       return err.message.replace("HTTP 400", "Bad request").trim();
     }
     return err.message || "An unexpected error occurred.";
   }

  // ---------------------------------------------------------------------------
  // Toast
  // ---------------------------------------------------------------------------
  function toast(message, type = "info") {
    const el = document.createElement("div");
    el.className = `toast toast-${type}`;
    el.textContent = message;
    toastContainer.appendChild(el);
    setTimeout(() => el.remove(), 4000);
  }

  // ---------------------------------------------------------------------------
  // Tab navigation
  // ---------------------------------------------------------------------------
  function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll(".tp-tab").forEach((el) => {
      el.classList.toggle("active", el.dataset.tab === tab);
    });
    stopPolling();
    stopLogRefresh();
    loadTab(tab);
  }

  document.querySelectorAll(".tp-tab").forEach((el) => {
    el.addEventListener("click", () => switchTab(el.dataset.tab));
  });

  async function loadTab(tab) {
    switch (tab) {
      case "datasets":
        await loadDatasets();
        break;
      case "training":
        await loadTraining();
        break;
      case "checkpoints":
        await loadCheckpoints();
        break;
      case "logs":
        await loadLogs();
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // Datasets
  // ---------------------------------------------------------------------------
   async function loadDatasets() {
     tpContent.innerHTML = `<div class="panel-card"><div class="loading-state"><div class="spinner"></div><p>Loading datasets…</p></div></div>`;
     try {
       const { datasets } = await api("training/datasets");
       const rows = datasets
        .map(
          (d) => `
        <tr>
          <td><strong>${escapeHtml(d.name)}</strong></td>
          <td>${d.size_mb} MB</td>
          <td>${d.lines}</td>
          <td>${d.modified}</td>
          <td>
            <button class="btn btn-sm btn-danger" data-delete="${escapeHtml(d.name)}">Delete</button>
            <button class="btn btn-sm btn-ghost" data-validate="${escapeHtml(d.name)}">Validate</button>
          </td>
        </tr>`
        )
        .join("");

      tpContent.innerHTML = `
        <div class="panel-card">
          <h3>Upload Dataset</h3>
          <div class="upload-zone" id="upload-zone">
            <h4>Drag & drop or click to upload</h4>
            <p>Supported: .txt, .json, .csv, .tsv, .parquet (max 50 MB)</p>
            <input type="file" id="file-input" accept=".txt,.json,.csv,.tsv,.parquet" style="display:none;">
          </div>
          <div id="upload-status" style="margin-top:10px;"></div>
        </div>
        <div class="panel-card">
          <h3>Datasets (${datasets.length})</h3>
          ${rows ? `<div style="overflow-x:auto;"><table class="data-table"><thead><tr><th>Name</th><th>Size</th><th>Lines</th><th>Modified</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table></div>` : "<div class='empty-state'><p>No datasets uploaded yet.</p></div>"}
        </div>
      `;

      // Upload zone
      const uploadZone = document.getElementById("upload-zone");
      const fileInput = document.getElementById("file-input");
      if (uploadZone && fileInput) {
        uploadZone.addEventListener("click", () => fileInput.click());
        uploadZone.addEventListener("dragover", (e) => {
          e.preventDefault();
          uploadZone.classList.add("dragover");
        });
        uploadZone.addEventListener("dragleave", () => {
          uploadZone.classList.remove("dragover");
        });
        uploadZone.addEventListener("drop", (e) => {
          e.preventDefault();
          uploadZone.classList.remove("dragover");
          const file = e.dataTransfer.files[0];
          if (file) uploadFile(file);
        });
        fileInput.addEventListener("change", () => {
          const file = fileInput.files[0];
          if (file) uploadFile(file);
        });
      }

      // Delete buttons
      tpContent.querySelectorAll("[data-delete]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const name = btn.dataset.delete;
          if (!confirm(`Delete dataset "${name}"?`)) return;
          try {
            await api(`training/datasets/${encodeURIComponent(name)}`, { method: "DELETE" });
            toast(`Dataset "${name}" deleted`, "success");
            loadDatasets();
          } catch (err) {
            toast(apiError(err), "error");
          }
        });
      });

      // Validate buttons
      tpContent.querySelectorAll("[data-validate]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const name = btn.dataset.validate;
          try {
            const result = await api(`training/datasets/${encodeURIComponent(name)}/validate`, { method: "POST" });
            toast(`Validation: ${result.valid_lines} valid lines, ${result.empty_lines} empty`, "info");
          } catch (err) {
            toast(apiError(err), "error");
          }
        });
      });
     } catch (err) {
       tpContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">${escapeHtml(apiError(err))}</p></div>`;
     }
  }

  async function uploadFile(file) {
    const statusEl = document.getElementById("upload-status");
    if (statusEl) statusEl.innerHTML = `<span style="color:var(--accent-yellow)">Uploading ${escapeHtml(file.name)}...</span>`;
    try {
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const result = await api("training/datasets", {
            method: "POST",
            body: JSON.stringify({ filename: file.name, content: Array.from(new Uint8Array(reader.result)).map(b => String.fromCharCode(b)).join("") }),
          });
          if (statusEl) statusEl.innerHTML = `<span style="color:var(--accent-green)">${result.name} uploaded (${result.size_mb} MB, ${result.lines} lines)</span>`;
          toast(`Dataset "${result.name}" uploaded`, "success");
          loadDatasets();
        } catch (err) {
          if (statusEl) statusEl.innerHTML = `<span style="color:var(--accent-red)">${escapeHtml(apiError(err))}</span>`;
          toast(apiError(err), "error");
        }
      };
      reader.readAsArrayBuffer(file);
    } catch (err) {
      if (statusEl) statusEl.innerHTML = `<span style="color:var(--accent-red)">${escapeHtml(apiError(err))}</span>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Training
  // ---------------------------------------------------------------------------
   async function loadTraining() {
     tpContent.innerHTML = `<div class="panel-card"><div class="loading-state"><div class="spinner"></div><p>Loading training status…</p></div></div>`;
     try {
       const status = await api("training/train/status");
      const isRunning = status.state === "running" || status.state === "stopping";
      const isPaused = status.state === "paused";
      const isDone = status.state === "finished" || status.state === "stopped" || status.state === "error";

      tpContent.innerHTML = `
        <div class="panel-card">
          <h3>Training Job</h3>
          <div class="resource-monitor">
            <div class="resource-item">
              <div class="resource-label">State</div>
              <div class="resource-value"><span class="badge badge-${status.state}">${status.state}</span></div>
            </div>
            <div class="resource-item">
              <div class="resource-label">Progress</div>
              <div class="resource-value">${status.progress}%</div>
            </div>
            <div class="resource-item">
              <div class="resource-label">Step</div>
              <div class="resource-value">${status.step} / ${status.config.max_steps || "—"}</div>
            </div>
            <div class="resource-item">
              <div class="resource-label">Loss</div>
              <div class="resource-value">${status.loss.toFixed(4)}</div>
            </div>
            <div class="resource-item">
              <div class="resource-label">Speed</div>
              <div class="resource-value">${status.tokens_per_sec.toFixed(1)} tok/s</div>
            </div>
            <div class="resource-item">
              <div class="resource-label">ETA</div>
              <div class="resource-value">${status.eta_formatted}</div>
            </div>
          </div>
          <div class="progress-bar"><div class="progress-fill ${status.progress > 80 ? 'green' : status.progress > 40 ? 'yellow' : ''}" style="width:${status.progress}%"></div></div>
          <div class="resource-monitor" style="margin-top:16px;">
            <div class="resource-item">
              <div class="resource-label">CPU Usage</div>
              <div class="resource-value">${status.cpu_usage}%</div>
            </div>
            <div class="resource-item">
              <div class="resource-label">GPU Usage</div>
              <div class="resource-value">${status.gpu_usage}%</div>
            </div>
            <div class="resource-item">
              <div class="resource-label">Memory</div>
              <div class="resource-value">${status.memory_usage}%</div>
            </div>
            <div class="resource-item">
              <div class="resource-label">Elapsed</div>
              <div class="resource-value">${formatUptime(status.elapsed)}</div>
            </div>
          </div>
          ${status.message ? `<p style="color:var(--text-secondary);font-size:0.82rem;margin-top:10px;">${escapeHtml(status.message)}</p>` : ""}
          <div class="form-actions" style="margin-top:14px;">
            ${isRunning ? '<button class="btn btn-danger" id="btn-stop-training">Stop</button>' : ""}
            ${isPaused ? '<button class="btn btn-primary" id="btn-resume-training">Resume</button>' : ""}
            ${!isRunning && !isPaused ? '<button class="btn btn-primary" id="btn-start-training">Start Training</button>' : ""}
          </div>
        </div>
        <div class="panel-card">
          <h3>Loss Graph</h3>
          <div class="loss-graph"><canvas id="loss-canvas"></canvas></div>
        </div>
        ${!isRunning && !isPaused ? renderTrainingForm(status.config) : ""}
      `;

      // Draw loss graph
      drawLossGraph(status.history || []);

      // Training control buttons
      const startBtn = document.getElementById("btn-start-training");
      if (startBtn) {
        startBtn.addEventListener("click", async () => {
          const body = {
            learning_rate: parseFloat(document.getElementById("trn-lr").value),
            max_steps: parseInt(document.getElementById("trn-steps").value),
            warmup_steps: parseInt(document.getElementById("trn-warmup").value),
            batch_size: parseInt(document.getElementById("trn-batch").value),
            max_seq_len: parseInt(document.getElementById("trn-seqlen").value),
            output_dir: document.getElementById("trn-outdir").value,
          };
          try {
            await api("training/train/start", { method: "POST", body: JSON.stringify(body) });
            toast("Training started", "success");
            startPolling();
            loadTraining();
          } catch (err) {
            toast(apiError(err), "error");
          }
        });
      }

      const stopBtn = document.getElementById("btn-stop-training");
      if (stopBtn) {
        stopBtn.addEventListener("click", async () => {
          try {
            await api("training/train/stop", { method: "POST" });
            toast("Stopping training…", "info");
            startPolling();
            loadTraining();
          } catch (err) {
            toast(apiError(err), "error");
          }
        });
      }

      const resumeBtn = document.getElementById("btn-resume-training");
      if (resumeBtn) {
        resumeBtn.addEventListener("click", async () => {
          try {
            await api("training/train/resume", { method: "POST" });
            toast("Resuming training…", "info");
            startPolling();
            loadTraining();
          } catch (err) {
            toast(apiError(err), "error");
          }
        });
      }

      if (isRunning || isPaused) startPolling();
     } catch (err) {
       tpContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">${escapeHtml(apiError(err))}</p></div>`;
     }
  }

  function renderTrainingForm(config) {
    return `
      <div class="panel-card">
        <h3>Training Configuration</h3>
        <div class="form-row">
          <div class="form-group"><label>Learning Rate</label><input id="trn-lr" type="number" step="0.0001" value="${config.learning_rate || 3e-4}"></div>
          <div class="form-group"><label>Max Steps</label><input id="trn-steps" type="number" min="1" value="${config.max_steps || 100}"></div>
          <div class="form-group"><label>Warmup Steps</label><input id="trn-warmup" type="number" min="0" value="${config.warmup_steps || 10}"></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Batch Size</label><input id="trn-batch" type="number" min="1" value="${config.batch_size || 2}"></div>
          <div class="form-group"><label>Max Seq Len</label><input id="trn-seqlen" type="number" min="16" value="${config.max_seq_len || 256}"></div>
          <div class="form-group"><label>Output Dir</label><input id="trn-outdir" value="${config.output_dir || "./checkpoints_demo"}"></div>
        </div>
      </div>
    `;
  }

  function drawLossGraph(history) {
    const canvas = document.getElementById("loss-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const data = history.map((h) => h.loss).filter((v) => v > 0);
    if (data.length < 2) {
      ctx.fillStyle = "#64748b";
      ctx.font = "14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No loss data yet", canvas.width / 2, canvas.height / 2);
      return;
    }

    const w = canvas.width;
    const h = canvas.height;
    const pad = 40;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;

    ctx.clearRect(0, 0, w, h);

    const minLoss = Math.min(...data) * 0.9;
    const maxLoss = Math.max(...data) * 1.1;
    const range = maxLoss - minLoss || 1;

    // Grid lines
    ctx.strokeStyle = "rgba(139, 92, 246, 0.1)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(w - pad, y);
      ctx.stroke();
    }

    // Loss line
    ctx.strokeStyle = "#8b5cf6";
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((loss, i) => {
      const x = pad + (i / (data.length - 1)) * plotW;
      const y = pad + plotH - ((loss - minLoss) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#64748b";
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.textAlign = "left";
    ctx.fillText(maxLoss.toFixed(3), 2, pad + 10);
    ctx.fillText(minLoss.toFixed(3), 2, h - pad);
  }

  // ---------------------------------------------------------------------------
  // Checkpoints
  // ---------------------------------------------------------------------------
   async function loadCheckpoints() {
     tpContent.innerHTML = `<div class="panel-card"><div class="loading-state"><div class="spinner"></div><p>Loading checkpoints…</p></div></div>`;
     try {
       const { checkpoints } = await api("training/checkpoints");
      const rows = checkpoints
        .map(
          (c) => `
        <tr>
          <td><strong>${escapeHtml(c.name)}</strong></td>
          <td>${c.size_mb} MB</td>
          <td>${c.modified}</td>
          <td>
            <button class="btn btn-sm btn-primary" data-load="${escapeHtml(c.name)}">Load</button>
            <button class="btn btn-sm btn-success" data-export="${escapeHtml(c.name)}">Export</button>
            <button class="btn btn-sm btn-danger" data-delete="${escapeHtml(c.name)}">Delete</button>
          </td>
        </tr>`
        )
        .join("");

      tpContent.innerHTML = `
        <div class="panel-card">
          <h3>Save Checkpoint</h3>
          <div class="form-row">
            <div class="form-group">
              <label>Checkpoint Name</label>
              <input id="ckpt-name" type="text" placeholder="my_checkpoint">
            </div>
          </div>
          <div class="form-actions">
            <button class="btn btn-primary" id="btn-save-ckpt">Save Checkpoint</button>
          </div>
        </div>
        <div class="panel-card">
          <h3>Checkpoints (${checkpoints.length})</h3>
          ${rows ? `<div style="overflow-x:auto;"><table class="data-table"><thead><tr><th>Name</th><th>Size</th><th>Modified</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table></div>` : "<div class='empty-state'><p>No checkpoints found.</p></div>"}
        </div>
      `;

      const saveBtn = document.getElementById("btn-save-ckpt");
      if (saveBtn) {
        saveBtn.addEventListener("click", async () => {
          const name = document.getElementById("ckpt-name").value.trim();
          if (!name) return toast("Checkpoint name is required", "error");
           try {
             const result = await api("training/checkpoints", { method: "POST", body: JSON.stringify({ name }) });
             toast(result.message || "Checkpoint saved", "success");
             loadCheckpoints();
           } catch (err) {
             toast(apiError(err), "error");
           }
         });
       }

       tpContent.querySelectorAll("[data-load]").forEach((btn) => {
         btn.addEventListener("click", async () => {
           const name = btn.dataset.load;
           if (!confirm(`Load checkpoint "${name}"?`)) return;
           try {
             const result = await api(`training/checkpoints/${encodeURIComponent(name)}/load`, { method: "POST" });
             toast(result.message || "Checkpoint loaded", "success");
           } catch (err) {
             toast(apiError(err), "error");
           }
         });
       });

       tpContent.querySelectorAll("[data-export]").forEach((btn) => {
         btn.addEventListener("click", async () => {
           const name = btn.dataset.export;
           try {
             const result = await api(`training/checkpoints/${encodeURIComponent(name)}/export`, { method: "POST", body: JSON.stringify({ format: "safetensors" }) });
             toast(result.message || "Exported", "success");
           } catch (err) {
             toast(apiError(err), "error");
           }
         });
       });

       tpContent.querySelectorAll("[data-delete]").forEach((btn) => {
         btn.addEventListener("click", async () => {
           const name = btn.dataset.delete;
           if (!confirm(`Delete checkpoint "${name}"?`)) return;
           try {
             await api(`training/checkpoints/${encodeURIComponent(name)}`, { method: "DELETE" });
             toast(`Checkpoint "${name}" deleted`, "success");
             loadCheckpoints();
           } catch (err) {
             toast(apiError(err), "error");
           }
         });
       });
     } catch (err) {
       tpContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">${escapeHtml(apiError(err))}</p></div>`;
     }
  }

  // ---------------------------------------------------------------------------
  // Logs
  // ---------------------------------------------------------------------------
   async function loadLogs() {
     tpContent.innerHTML = `<div class="panel-card"><div class="loading-state"><div class="spinner"></div><p>Loading logs…</p></div></div>`;
     try {
       const { logs, total } = await api("training/logs?limit=500");
      const entries = logs
        .map(
          (l) =>
            `<div class="log-entry"><span class="log-level-${l.level}">[${l.time}] [${l.level}] ${escapeHtml(l.message)}</span></div>`
        )
        .join("");

      tpContent.innerHTML = `
        <div class="panel-card">
          <div class="action-bar">
            <button class="btn btn-primary" id="btn-refresh-logs">Refresh</button>
            <button class="btn btn-danger" id="btn-clear-logs">Clear Logs</button>
            <input type="text" class="search-input" id="log-search" placeholder="Search logs...">
            <select id="log-level-filter">
              <option value="">All Levels</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
              <option value="CRITICAL">CRITICAL</option>
            </select>
          </div>
          <div class="log-viewer" id="log-viewer">${entries || "<span style='color:var(--text-dim)'>No logs yet.</span>"}</div>
          <div style="margin-top:8px;font-size:0.75rem;color:var(--text-dim);">Total: ${total} entries</div>
        </div>
      `;

      document.getElementById("btn-refresh-logs").addEventListener("click", loadLogs);
       document.getElementById("btn-clear-logs").addEventListener("click", async () => {
         if (!confirm("Clear all logs?")) return;
         try {
           await api("training/logs", { method: "DELETE" });
           toast("Logs cleared", "info");
           loadLogs();
         } catch (err) {
           toast(apiError(err), "error");
         }
       });

      const searchInput = document.getElementById("log-search");
      const levelFilter = document.getElementById("log-level-filter");
      const refreshLogs = async () => {
        const level = levelFilter ? levelFilter.value : "";
        const search = searchInput ? searchInput.value : "";
        try {
          const result = await api("training/logs", {
            method: "POST",
            body: JSON.stringify({ level, search, limit: 500 }),
          });
          const logViewer = document.getElementById("log-viewer");
          if (logViewer) {
            const entries = result.logs
              .map(
                (l) =>
                  `<div class="log-entry"><span class="log-level-${l.level}">[${l.time}] [${l.level}] ${escapeHtml(l.message)}</span></div>`
              )
              .join("");
            logViewer.innerHTML = entries || "<span style='color:var(--text-dim)'>No logs match your filter.</span>";
          }
        } catch (_) { /* ignore */ }
      };

      if (searchInput) searchInput.addEventListener("input", refreshLogs);
      if (levelFilter) levelFilter.addEventListener("change", refreshLogs);

      stopLogRefresh();
      logAutoRefresh = setInterval(loadLogs, 3000);
     } catch (err) {
       tpContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">${escapeHtml(apiError(err))}</p></div>`;
     }
  }

  // ---------------------------------------------------------------------------
  // Polling helpers
  // ---------------------------------------------------------------------------
  function startPolling() {
    stopPolling();
    pollingInterval = setInterval(async () => {
      try {
        if (currentTab === "training") {
          await loadTraining();
        }
      } catch (_) { /* ignore */ }
    }, 3000);
  }

  function stopPolling() {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
  }

  function stopLogRefresh() {
    if (logAutoRefresh) {
      clearInterval(logAutoRefresh);
      logAutoRefresh = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Utility
  // ---------------------------------------------------------------------------
  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function formatUptime(seconds) {
    if (!seconds || seconds < 0) return "0s";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    initTheme();
    loadTab("datasets");
  }

  init();
})();