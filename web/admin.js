/**
 * Netcradus Admin Dashboard — Frontend Controller.
 *
 * Features:
 * - Secure login with token-based auth (Bearer).
 * - SPA-style section navigation (Dashboard, Users, LLM, Settings, Training, Logs).
 * - Real-time training progress polling.
 * - Auto-refreshing logs viewer.
 * - User CRUD with confirmation dialogs.
 * - Responsive sidebar with hamburger toggle.
 * - Toast notifications for success/error/info.
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // DOM references
  // ---------------------------------------------------------------------------
  const loginView = document.getElementById("login-view");
  const appView = document.getElementById("app-view");
  const loginForm = document.getElementById("login-form");
  const loginError = document.getElementById("login-error");
  const loginUsername = document.getElementById("login-username");
  const loginPassword = document.getElementById("login-password");
  const btnLogout = document.getElementById("btn-logout");
  const sidebar = document.getElementById("admin-sidebar");
  const btnHamburger = document.getElementById("btn-hamburger");
  const sectionTitle = document.getElementById("section-title");
  const adminContent = document.getElementById("admin-content");
  const toastContainer = document.getElementById("toast-container");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let token = localStorage.getItem("netcradus_admin_token") || null;
  let user = null;
  let currentSection = "dashboard";
  let pollingInterval = null;
  let logAutoRefresh = null;

  // ---------------------------------------------------------------------------
  // API helper
  // ---------------------------------------------------------------------------
  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await fetch(`/api/${path}`, { headers, ...options });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
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
  // Auth
  // ---------------------------------------------------------------------------
  function showLogin() {
    loginView.style.display = "flex";
    appView.style.display = "none";
    token = null;
    user = null;
    localStorage.removeItem("netcradus_admin_token");
  }

  function showApp() {
    loginView.style.display = "none";
    appView.style.display = "flex";
    document.getElementById("sidebar-username").textContent = user.username;
    document.getElementById("sidebar-role").textContent =
      user.role.charAt(0).toUpperCase() + user.role.slice(1);
    document.getElementById("sidebar-user-avatar").textContent =
      user.username.charAt(0).toUpperCase();
  }

  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    loginError.style.display = "none";
    try {
      const data = await api("auth/login", {
        method: "POST",
        body: JSON.stringify({
          username: loginUsername.value.trim(),
          password: loginPassword.value,
        }),
      });
      token = data.token;
      user = data.user;
      localStorage.setItem("netcradus_admin_token", token);
      showApp();
      navigate("dashboard");
      toast("Signed in successfully", "success");
    } catch (err) {
      loginError.textContent = err.message;
      loginError.style.display = "block";
    }
  });

  btnLogout.addEventListener("click", async () => {
    try {
      await api("auth/logout", { method: "POST" });
    } catch (_) { /* ignore */ }
    token = null;
    user = null;
    localStorage.removeItem("netcradus_admin_token");
    stopPolling();
    stopLogRefresh();
    showLogin();
  });

  // ---------------------------------------------------------------------------
  // Navigation
  // ---------------------------------------------------------------------------
  function navigate(section) {
    currentSection = section;
    document.querySelectorAll(".nav-item").forEach((el) => {
      el.classList.toggle("active", el.dataset.section === section);
    });
    sectionTitle.textContent =
      section.charAt(0).toUpperCase() + section.slice(1);
    document.querySelectorAll(".section-panel").forEach((el) => {
      el.classList.toggle("active", el.id === `panel-${section}`);
    });
    if (window.innerWidth <= 900) sidebar.classList.remove("open");
    loadSection(section);
  }

  document.querySelectorAll(".nav-item").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      navigate(el.dataset.section);
    });
  });

  // ---------------------------------------------------------------------------
  // Section loaders
  // ---------------------------------------------------------------------------
  async function loadSection(section) {
    stopPolling();
    stopLogRefresh();
    switch (section) {
      case "dashboard":
        await loadDashboard();
        break;
      case "users":
        await loadUsers();
        break;
      case "llm":
        await loadLLM();
        break;
      case "settings":
        await loadSettings();
        break;
      case "training":
        await loadTraining();
        break;
      case "logs":
        await loadLogs();
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // Dashboard
  // ---------------------------------------------------------------------------
  async function loadDashboard() {
    try {
      const d = await api("admin/dashboard");
      const uptime = formatUptime(d.uptime);
      adminContent.innerHTML = `
        <div class="cards-grid">
          <div class="stat-card accent">
            <div class="stat-label">Model Status</div>
            <div class="stat-value">${d.pipeline_loaded ? "Loaded" : "Fallback"}</div>
            <div class="stat-sub">${d.model_name} | ${d.device}</div>
          </div>
          <div class="stat-card green">
            <div class="stat-label">Parameters</div>
            <div class="stat-value">${(d.params / 1e6).toFixed(1)}M</div>
            <div class="stat-sub">Vocab: ${d.vocab_size}</div>
          </div>
          <div class="stat-card cyan">
            <div class="stat-label">Users</div>
            <div class="stat-value">${d.users.length}</div>
            <div class="stat-sub">${d.users.filter((u) => u.role === "admin").length} admin(s)</div>
          </div>
          <div class="stat-card yellow">
            <div class="stat-label">Checkpoints</div>
            <div class="stat-value">${d.checkpoints}</div>
            <div class="stat-sub">Loaded: ${d.loaded_checkpoint || "none"}</div>
          </div>
          <div class="stat-card accent">
            <div class="stat-label">Uptime</div>
            <div class="stat-value">${uptime}</div>
            <div class="stat-sub">Requests served: ${d.requests}</div>
          </div>
          <div class="stat-card green">
            <div class="stat-label">Training</div>
            <div class="stat-value">${d.training_state || "idle"}</div>
            <div class="stat-sub">${d.training_state === "running" ? "Active job" : "No active job"}</div>
          </div>
        </div>
        <div class="panel-card">
          <h3>System Info</h3>
          <pre style="font-family:var(--mono);font-size:0.78rem;color:var(--text-secondary);white-space:pre-wrap;">${JSON.stringify(d.model_config, null, 2) || "N/A"}</pre>
        </div>
      `;
    } catch (err) {
      adminContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load dashboard: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Users
  // ---------------------------------------------------------------------------
  async function loadUsers() {
    try {
      const { users } = await api("admin/users");
      let rows = users
        .map(
          (u) => `
        <tr>
          <td><strong>${escapeHtml(u.username)}</strong></td>
          <td><span class="badge badge-${u.role}">${u.role}</span></td>
          <td>${u.created_at ? new Date(u.created_at * 1000).toLocaleString() : "—"}</td>
          <td>
            <button class="btn btn-sm btn-primary" data-edit="${escapeHtml(u.username)}">Edit</button>
            <button class="btn btn-sm btn-danger" data-delete="${escapeHtml(u.username)}">Delete</button>
            <button class="btn btn-sm btn-ghost" data-password="${escapeHtml(u.username)}">Password</button>
          </td>
        </tr>`
        )
        .join("");

      adminContent.innerHTML = `
        <div class="panel-card">
          <h3>Add User</h3>
          <div class="form-row">
            <div class="form-group"><label>Username</label><input id="new-username" placeholder="user"></div>
            <div class="form-group"><label>Password</label><input id="new-password" type="password" placeholder="••••••••"></div>
            <div class="form-group">
              <label>Role</label>
              <select id="new-role"><option value="user">User</option><option value="admin">Admin</option></select>
            </div>
          </div>
          <div class="form-actions">
            <button class="btn btn-primary" id="btn-create-user">Create User</button>
          </div>
        </div>
        <div class="panel-card">
          <h3>Users (${users.length})</h3>
          <div style="overflow-x:auto;">
            <table class="data-table">
              <thead><tr><th>Username</th><th>Role</th><th>Created</th><th>Action</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>
      `;

      document.getElementById("btn-create-user").addEventListener("click", async () => {
        const username = document.getElementById("new-username").value.trim();
        const password = document.getElementById("new-password").value;
        const role = document.getElementById("new-role").value;
        if (!username || !password) return toast("Username and password required", "error");
        try {
          await api("admin/users", { method: "POST", body: JSON.stringify({ username, password, role }) });
          toast("User created", "success");
          loadUsers();
        } catch (err) {
          toast(err.message, "error");
        }
      });

      adminContent.querySelectorAll("[data-delete]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const username = btn.dataset.delete;
          if (!confirm(`Delete user "${username}"?`)) return;
          try {
            await api(`admin/users/${encodeURIComponent(username)}`, { method: "DELETE" });
            toast("User deleted", "success");
            loadUsers();
          } catch (err) {
            toast(err.message, "error");
          }
        });
      });

      adminContent.querySelectorAll("[data-edit]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const username = btn.dataset.edit;
          const newRole = prompt(
            `Enter new role for "${username}" (admin/user):`,
            "user"
          );
          if (!newRole) return;
          if (newRole !== "admin" && newRole !== "user") {
            return toast("Role must be 'admin' or 'user'", "error");
          }
          try {
            await api(`admin/users/${encodeURIComponent(username)}`, {
              method: "PUT",
              body: JSON.stringify({ role: newRole }),
            });
            toast(`User "${username}" updated`, "success");
            loadUsers();
          } catch (err) {
            toast(err.message, "error");
          }
        });
      });

      adminContent.querySelectorAll("[data-password]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const username = btn.dataset.password;
          const newPassword = prompt(`Enter new password for "${username}":`);
          if (!newPassword) return;
          if (newPassword.length < 6) {
            return toast("Password must be at least 6 characters", "error");
          }
          try {
            await api(`admin/users/${encodeURIComponent(username)}/password`, {
              method: "POST",
              body: JSON.stringify({ password: newPassword }),
            });
            toast(`Password updated for "${username}"`, "success");
          } catch (err) {
            toast(err.message, "error");
          }
        });
      });
    } catch (err) {
      adminContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load users: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // LLM Management
  // ---------------------------------------------------------------------------
  async function loadLLM() {
    try {
      const info = await api("admin/llm");
      const checkpoints = await api("admin/checkpoints");
      const ckptRows = checkpoints.checkpoints
        .map(
          (c) => `
        <tr>
          <td>${escapeHtml(c.name)}</td>
          <td>${c.size_mb} MB</td>
          <td>${c.modified}</td>
          <td>
            <button class="btn btn-sm btn-primary" data-load="${escapeHtml(c.name)}">Load</button>
            <button class="btn btn-sm btn-danger" data-delete="${escapeHtml(c.name)}">Delete</button>
          </td>
        </tr>`
        )
        .join("");

      adminContent.innerHTML = `
        <div class="panel-card">
          <h3>Model Status</h3>
          <div class="form-row">
            <div class="form-group">
              <label>Pipeline</label>
              <span class="badge ${info.pipeline_loaded ? "badge-running" : "badge-idle"}">${info.pipeline_loaded ? "Loaded" : "Fallback"}</span>
            </div>
            <div class="form-group">
              <label>Device</label>
              <span>${info.device}</span>
            </div>
            <div class="form-group">
              <label>Loaded Checkpoint</label>
              <span>${info.loaded_checkpoint || "—"}</span>
            </div>
          </div>
          ${info.pipeline_loaded ? `<p style="color:var(--text-secondary);font-size:0.82rem;">Params: ${(info.params / 1e6).toFixed(1)}M | Vocab: ${info.vocab_size} | Arch: ${info.architecture}</p>` : ""}
        </div>
        <div class="panel-card">
          <div class="form-actions" style="margin-bottom:14px;">
            <button class="btn btn-primary" id="btn-unload">Unload Model</button>
          </div>
          <h3>Checkpoints (${checkpoints.checkpoints.length})</h3>
          ${ckptRows ? `<div style="overflow-x:auto;"><table class="data-table"><thead><tr><th>Name</th><th>Size</th><th>Modified</th><th>Actions</th></tr></thead><tbody>${ckptRows}</tbody></table></div>` : "<p style='color:var(--text-dim)'>No checkpoints found.</p>"}
        </div>
      `;

      const unloadBtn = document.getElementById("btn-unload");
      if (unloadBtn) {
        unloadBtn.addEventListener("click", async () => {
          try {
            await api("admin/llm/unload", { method: "POST" });
            toast("Model unloaded", "info");
            loadLLM();
          } catch (err) {
            toast(err.message, "error");
          }
        });
      }

      adminContent.querySelectorAll("[data-load]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          try {
            const res = await api("admin/llm/load", {
              method: "POST",
              body: JSON.stringify({ checkpoint: btn.dataset.load }),
            });
            toast(res.message || "Checkpoint loaded", "success");
            loadLLM();
          } catch (err) {
            toast(err.message, "error");
          }
        });
      });

      adminContent.querySelectorAll("[data-delete]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          if (!confirm(`Delete checkpoint "${btn.dataset.delete}"?`)) return;
          try {
            await api(`admin/checkpoints/${encodeURIComponent(btn.dataset.delete)}`, { method: "DELETE" });
            toast("Checkpoint deleted", "success");
            loadLLM();
          } catch (err) {
            toast(err.message, "error");
          }
        });
      });
    } catch (err) {
      adminContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load LLM info: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Model Settings
  // ---------------------------------------------------------------------------
  async function loadSettings() {
    try {
      const { settings } = await api("admin/settings");
      adminContent.innerHTML = `
        <div class="panel-card">
          <h3>Model Settings</h3>
          <div class="form-row">
            <div class="form-group">
              <label>Default Temperature</label>
              <input type="number" id="set-temp" step="0.1" min="0" max="2" value="${settings.default_temperature}">
            </div>
            <div class="form-group">
              <label>Default Max Tokens</label>
              <input type="number" id="set-max-tokens" step="1" min="1" max="${settings.max_tokens_limit}" value="${settings.default_max_tokens}">
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label>Default Top-K</label>
              <input type="number" id="set-top-k" step="1" min="1" value="${settings.default_top_k}">
            </div>
            <div class="form-group">
              <label>Default Top-P</label>
              <input type="number" id="set-top-p" step="0.05" min="0" max="1" value="${settings.default_top_p}">
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label>Default Persona</label>
              <select id="set-persona">
                <option value="general" ${settings.default_persona === "general" ? "selected" : ""}>General</option>
                <option value="code" ${settings.default_persona === "code" ? "selected" : ""}>Code</option>
                <option value="reasoning" ${settings.default_persona === "reasoning" ? "selected" : ""}>Reasoning</option>
                <option value="creative" ${settings.default_persona === "creative" ? "selected" : ""}>Creative</option>
              </select>
            </div>
            <div class="form-group">
              <label>Stream Enabled</label>
              <select id="set-stream">
                <option value="true" ${settings.stream_enabled ? "selected" : ""}>Yes</option>
                <option value="false" ${!settings.stream_enabled ? "selected" : ""}>No</option>
              </select>
            </div>
          </div>
          <div class="form-actions">
            <button class="btn btn-primary" id="btn-save-settings">Save Settings</button>
          </div>
        </div>
      `;

      document.getElementById("btn-save-settings").addEventListener("click", async () => {
        const body = {
          default_temperature: parseFloat(document.getElementById("set-temp").value),
          default_max_tokens: parseInt(document.getElementById("set-max-tokens").value),
          default_top_k: parseInt(document.getElementById("set-top-k").value),
          default_top_p: parseFloat(document.getElementById("set-top-p").value),
          default_persona: document.getElementById("set-persona").value,
          stream_enabled: document.getElementById("set-stream").value === "true",
        };
        try {
          const res = await api("admin/settings", { method: "POST", body: JSON.stringify(body) });
          toast("Settings saved", "success");
          loadSettings();
        } catch (err) {
          toast(err.message, "error");
        }
      });
    } catch (err) {
      adminContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load settings: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Training
  // ---------------------------------------------------------------------------
  async function loadTraining() {
    try {
      const status = await api("admin/training/status");
      const isRunning = status.state === "running" || status.state === "stopping";
      const isDone = status.state === "finished" || status.state === "stopped" || status.state === "error";

      adminContent.innerHTML = `
        <div class="panel-card">
          <h3>Training Job</h3>
          <div class="form-row">
            <div class="form-group">
              <label>State</label>
              <span class="badge badge-${status.state}">${status.state}</span>
            </div>
            <div class="form-group">
              <label>Step</label>
              <span>${status.step} / ${status.config.max_steps || "—"}</span>
            </div>
            <div class="form-group">
              <label>Loss</label>
              <span>${status.loss.toFixed(4)}</span>
            </div>
            <div class="form-group">
              <label>Speed</label>
              <span>${status.tokens_per_sec.toFixed(1)} tok/s</span>
            </div>
            <div class="form-group">
              <label>Elapsed</label>
              <span>${formatUptime(status.elapsed)}</span>
            </div>
          </div>
          ${status.message ? `<p style="color:var(--text-secondary);font-size:0.82rem;margin-top:8px;">${escapeHtml(status.message)}</p>` : ""}
          <div class="progress-bar"><div class="progress-fill" style="width:${status.config.max_steps ? Math.min(100, (status.step / status.config.max_steps) * 100) : 0}%"></div></div>
          <div class="form-actions" style="margin-top:14px;">
            ${isRunning ? '<button class="btn btn-danger" id="btn-stop-training">Stop Training</button>' : ""}
            ${!isRunning ? '<button class="btn btn-primary" id="btn-start-training">Start Training</button>' : ""}
          </div>
        </div>
        ${!isRunning ? renderTrainingForm(status.config) : ""}
      `;

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
            await api("admin/training/start", { method: "POST", body: JSON.stringify(body) });
            toast("Training started", "success");
            startPolling();
            loadTraining();
          } catch (err) {
            toast(err.message, "error");
          }
        });
      }

      const stopBtn = document.getElementById("btn-stop-training");
      if (stopBtn) {
        stopBtn.addEventListener("click", async () => {
          try {
            await api("admin/training/stop", { method: "POST" });
            toast("Stopping training…", "info");
            startPolling();
            loadTraining();
          } catch (err) {
            toast(err.message, "error");
          }
        });
      }

      if (isRunning) startPolling();
    } catch (err) {
      adminContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load training status: ${escapeHtml(err.message)}</p></div>`;
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

  // ---------------------------------------------------------------------------
  // Logs
  // ---------------------------------------------------------------------------
  async function loadLogs() {
    try {
      const { logs } = await api("admin/logs?limit=500");
      const entries = logs
        .map(
          (l) =>
            `<div class="log-entry"><span class="log-level-${l.level}">[${l.time}] [${l.level}] ${escapeHtml(l.message)}</span></div>`
        )
        .join("");

      adminContent.innerHTML = `
        <div class="panel-card">
          <div class="form-actions" style="margin-bottom:14px;">
            <button class="btn btn-primary" id="btn-refresh-logs">Refresh</button>
            <button class="btn btn-danger" id="btn-clear-logs">Clear Logs</button>
          </div>
          <div class="log-viewer" id="log-viewer">${entries || "<span style='color:var(--text-dim)'>No logs yet.</span>"}</div>
        </div>
      `;

      document.getElementById("btn-refresh-logs").addEventListener("click", loadLogs);
      document.getElementById("btn-clear-logs").addEventListener("click", async () => {
        if (!confirm("Clear all logs?")) return;
        try {
          await api("admin/logs", { method: "DELETE" });
          toast("Logs cleared", "info");
          loadLogs();
        } catch (err) {
          toast(err.message, "error");
        }
      });

      stopLogRefresh();
      logAutoRefresh = setInterval(loadLogs, 3000);
    } catch (err) {
      adminContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load logs: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Polling helpers
  // ---------------------------------------------------------------------------
  function startPolling() {
    stopPolling();
    pollingInterval = setInterval(async () => {
      try {
        const status = await api("admin/training/status");
        if (status.state === "finished" || status.state === "stopped" || status.state === "error") {
          stopPolling();
          toast(`Training ${status.state}: ${status.message}`, status.state === "error" ? "error" : "info");
        }
        if (currentSection === "training") loadTraining();
      } catch (_) { /* ignore */ }
    }, 2000);
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
  // Sidebar toggle (mobile)
  // ---------------------------------------------------------------------------
  btnHamburger.addEventListener("click", () => {
    sidebar.classList.toggle("open");
  });

  document.addEventListener("click", (e) => {
    if (
      window.innerWidth <= 900 &&
      sidebar.classList.contains("open") &&
      !sidebar.contains(e.target) &&
      e.target !== btnHamburger &&
      !btnHamburger.contains(e.target)
    ) {
      sidebar.classList.remove("open");
    }
  });

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    if (token) {
      api("auth/me")
        .then((data) => {
          user = data.user;
          showApp();
          navigate("dashboard");
        })
        .catch(() => {
          showLogin();
        });
    } else {
      showLogin();
    }
  }

  init();

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
})();