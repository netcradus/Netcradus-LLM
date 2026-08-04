/**
 * Netcradus User Panel — Frontend Controller.
 *
 * Features:
 * - Login / Register / Logout with token-based auth.
 * - SPA navigation (Dashboard, Chat, Profile, Settings).
 * - Real-time chat with Netcradus LLM (SSE streaming).
 * - Chat history sidebar with auto-refresh.
 * - Profile editing and password change.
 * - Dark / Light mode toggle with persistence.
 * - Responsive sidebar with hamburger toggle.
 * - Toast notifications.
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // DOM references
  // ---------------------------------------------------------------------------
  const loginView = document.getElementById("login-view");
  const registerView = document.getElementById("register-view");
  const appView = document.getElementById("app-view");
  const loginForm = document.getElementById("login-form");
  const loginError = document.getElementById("login-error");
  const loginUsername = document.getElementById("login-username");
  const loginPassword = document.getElementById("login-password");
  const registerForm = document.getElementById("register-form");
  const registerError = document.getElementById("register-error");
  const btnLogout = document.getElementById("btn-logout");
  const btnThemeToggle = document.getElementById("btn-theme-toggle");
  const sidebar = document.getElementById("user-sidebar");
  const btnHamburger = document.getElementById("btn-hamburger");
  const sectionTitle = document.getElementById("section-title");
  const userContent = document.getElementById("user-content");
  const toastContainer = document.getElementById("toast-container");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const themeIcon = document.getElementById("theme-icon");

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let token = localStorage.getItem("netcradus_user_token") || null;
  let user = null;
  let currentSection = "dashboard";
  let pollingInterval = null;
  let currentChatId = null;

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
    if (theme === "dark") {
      themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    } else {
      themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
    }
  }

  btnThemeToggle.addEventListener("click", toggleTheme);

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
    registerView.style.display = "none";
    appView.style.display = "none";
    token = null;
    user = null;
    localStorage.removeItem("netcradus_user_token");
  }

  function showRegister() {
    loginView.style.display = "none";
    registerView.style.display = "flex";
    appView.style.display = "none";
  }

  function showApp() {
    loginView.style.display = "none";
    registerView.style.display = "none";
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
      const data = await api("user/auth/login", {
        method: "POST",
        body: JSON.stringify({
          username: loginUsername.value.trim(),
          password: loginPassword.value,
        }),
      });
      token = data.token;
      user = data.user;
      localStorage.setItem("netcradus_user_token", token);
      showApp();
      navigate("dashboard");
      toast("Signed in successfully", "success");
    } catch (err) {
      loginError.textContent = err.message;
      loginError.style.display = "block";
    }
  });

  registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    registerError.style.display = "none";
    try {
      const data = await api("user/auth/register", {
        method: "POST",
        body: JSON.stringify({
          username: document.getElementById("reg-username").value.trim(),
          display_name: document.getElementById("reg-displayname").value.trim(),
          password: document.getElementById("reg-password").value,
        }),
      });
      token = data.token;
      user = data.user;
      localStorage.setItem("netcradus_user_token", token);
      showApp();
      navigate("dashboard");
      toast("Account created successfully", "success");
    } catch (err) {
      registerError.textContent = err.message;
      registerError.style.display = "block";
    }
  });

  document.getElementById("link-register").addEventListener("click", (e) => {
    e.preventDefault();
    showRegister();
  });

  document.getElementById("link-login").addEventListener("click", (e) => {
    e.preventDefault();
    showLogin();
  });

  btnLogout.addEventListener("click", async () => {
    try {
      await api("user/auth/logout", { method: "POST" });
    } catch (_) { /* ignore */ }
    token = null;
    user = null;
    localStorage.removeItem("netcradus_user_token");
    stopPolling();
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
    switch (section) {
      case "dashboard":
        await loadDashboard();
        break;
      case "chat":
        await loadChat();
        break;
      case "profile":
        await loadProfile();
        break;
      case "settings":
        await loadSettings();
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // Dashboard
  // ---------------------------------------------------------------------------
  async function loadDashboard() {
    try {
      const d = await api("user/dashboard");
      userContent.innerHTML = `
        <div class="cards-grid">
          <div class="stat-card accent">
            <div class="stat-label">Welcome</div>
            <div class="stat-value">${escapeHtml(d.user.display_name || d.user.username)}</div>
            <div class="stat-sub">Member since ${formatDate(d.user.created_at)}</div>
          </div>
          <div class="stat-card green">
            <div class="stat-label">Chats</div>
            <div class="stat-value">${d.chat_count}</div>
            <div class="stat-sub">Conversations</div>
          </div>
          <div class="stat-card cyan">
            <div class="stat-label">Role</div>
            <div class="stat-value">${d.user.role}</div>
            <div class="stat-sub">${d.user.role === 'admin' ? 'Full access' : 'Standard access'}</div>
          </div>
        </div>
        <div class="panel-card">
          <h3>Quick Actions</h3>
          <div class="form-actions">
            <button class="btn btn-primary" id="btn-new-chat">New Chat</button>
            <button class="btn" id="btn-edit-profile">Edit Profile</button>
          </div>
        </div>
      `;
      const newChatBtn = document.getElementById("btn-new-chat");
      if (newChatBtn) {
        newChatBtn.addEventListener("click", () => navigate("chat"));
      }
      const editProfileBtn = document.getElementById("btn-edit-profile");
      if (editProfileBtn) {
        editProfileBtn.addEventListener("click", () => navigate("profile"));
      }
    } catch (err) {
      userContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load dashboard: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Chat
  // ---------------------------------------------------------------------------
  async function loadChat() {
    try {
      const { history } = await api("user/chat-history");
      const chatItems = history
        .map(
          (c) => `
        <div class="chat-sidebar-item" data-chat-id="${c.id}">
          <div class="chat-title">${escapeHtml(c.messages[0]?.content || "New Chat").substring(0, 40)}...</div>
          <div class="chat-date">${formatDate(c.created_at)}</div>
        </div>`
        )
        .join("");

      userContent.innerHTML = `
        <div class="chat-layout">
          <div class="chat-sidebar">
            <div class="chat-sidebar-header">
              <h3>Chats</h3>
              <button class="btn btn-sm btn-primary" id="btn-new-chat-sidebar">+ New</button>
            </div>
            <div class="chat-sidebar-list" id="chat-list">
              ${chatItems || '<div class="empty-state"><p>No chats yet</p></div>'}
            </div>
          </div>
          <div class="chat-main">
            <div class="chat-messages" id="chat-messages">
              <div class="empty-state">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                <p>Start a conversation with Netcradus LLM</p>
              </div>
            </div>
            <div class="chat-input-bar">
              <input type="text" id="chat-input" placeholder="Type your message..." autocomplete="off">
              <button class="btn btn-primary" id="btn-send">Send</button>
            </div>
          </div>
        </div>
      `;

      // New chat button
      const newChatBtn = document.getElementById("btn-new-chat-sidebar");
      if (newChatBtn) {
        newChatBtn.addEventListener("click", () => startNewChat());
      }

      // Send message
      const sendBtn = document.getElementById("btn-send");
      const chatInput = document.getElementById("chat-input");
      if (sendBtn && chatInput) {
        const sendMessage = async () => {
          const message = chatInput.value.trim();
          if (!message) return;
          chatInput.value = "";
          await sendChatMessage(message);
        };
        sendBtn.addEventListener("click", sendMessage);
        chatInput.addEventListener("keydown", (e) => {
          if (e.key === "Enter") sendMessage();
        });
      }

      // Chat history items
      userContent.querySelectorAll("[data-chat-id]").forEach((item) => {
        item.addEventListener("click", () => {
          const chatId = parseInt(item.dataset.chatId);
          loadChatHistory(chatId);
        });
      });
    } catch (err) {
      userContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load chat: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  async function startNewChat() {
    currentChatId = null;
    const messagesContainer = document.getElementById("chat-messages");
    if (messagesContainer) {
      messagesContainer.innerHTML = `
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          <p>Start a new conversation</p>
        </div>
      `;
    }
    const chatInput = document.getElementById("chat-input");
    if (chatInput) chatInput.focus();
  }

  async function sendChatMessage(message) {
    const messagesContainer = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    if (!messagesContainer) return;

    // Add user message
    const userMsg = document.createElement("div");
    userMsg.className = "chat-message user";
    userMsg.innerHTML = `${escapeHtml(message)}<div class="msg-time">${new Date().toLocaleTimeString()}</div>`;
    messagesContainer.appendChild(userMsg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Show typing indicator
    const typing = document.createElement("div");
    typing.className = "typing-indicator";
    typing.innerHTML = "<span></span><span></span><span></span>";
    messagesContainer.appendChild(typing);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    if (chatInput) chatInput.disabled = true;

    try {
      const res = await api("user/chat", {
        method: "POST",
        body: JSON.stringify({ message, chat_id: currentChatId }),
      });

      typing.remove();

      // Add assistant response
      const assistantMsg = document.createElement("div");
      assistantMsg.className = "chat-message assistant";
      assistantMsg.innerHTML = `${escapeHtml(res.response)}<div class="msg-time">${new Date().toLocaleTimeString()}</div>`;
      messagesContainer.appendChild(assistantMsg);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;

      currentChatId = res.chat_id || currentChatId;

      // Save to chat history
      await api("user/chat-history", {
        method: "POST",
        body: JSON.stringify({
          chat_id: currentChatId,
          messages: [
            { role: "user", content: message },
            { role: "assistant", content: res.response },
          ],
        }),
      });
    } catch (err) {
      typing.remove();
      const errorMsg = document.createElement("div");
      errorMsg.className = "chat-message assistant";
      errorMsg.innerHTML = `<span style="color:var(--accent-red)">Error: ${escapeHtml(err.message)}</span>`;
      messagesContainer.appendChild(errorMsg);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    if (chatInput) {
      chatInput.disabled = false;
      chatInput.focus();
    }
  }

  async function loadChatHistory(chatId) {
    try {
      const { history } = await api(`user/chat-history/${chatId}`);
      if (history && history.length > 0) {
        currentChatId = chatId;
        const messagesContainer = document.getElementById("chat-messages");
        if (messagesContainer) {
          const chat = history[0];
          messagesContainer.innerHTML = chat.messages
            .map(
              (m) => `
            <div class="chat-message ${m.role}">
              ${escapeHtml(m.content)}
              <div class="msg-time">${new Date(m.created_at || Date.now()).toLocaleTimeString()}</div>
            </div>
          `
            )
            .join("");
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
      }
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // ---------------------------------------------------------------------------
  // Profile
  // ---------------------------------------------------------------------------
  async function loadProfile() {
    try {
      const { user: profile } = await api("user/profile");
      userContent.innerHTML = `
        <div class="panel-card">
          <h3>Profile</h3>
          <div class="form-row">
            <div class="form-group">
              <label>Username</label>
              <input type="text" value="${escapeHtml(profile.username)}" disabled>
            </div>
            <div class="form-group">
              <label>Display Name</label>
              <input type="text" id="prof-displayname" value="${escapeHtml(profile.display_name || "")}" placeholder="Your display name">
            </div>
          </div>
          <div class="form-group">
            <label>Bio</label>
            <textarea id="prof-bio" rows="3" placeholder="Tell us about yourself">${escapeHtml(profile.bio || "")}</textarea>
          </div>
          <div class="form-group">
            <label>Avatar Color</label>
            <input type="color" id="prof-avatar-color" value="${profile.avatar_color || '#8b5cf6'}">
          </div>
          <div class="form-actions">
            <button class="btn btn-primary" id="btn-save-profile">Save Profile</button>
          </div>
        </div>
      `;

      document.getElementById("btn-save-profile").addEventListener("click", async () => {
        try {
          await api("user/profile", {
            method: "PUT",
            body: JSON.stringify({
              display_name: document.getElementById("prof-displayname").value.trim(),
              bio: document.getElementById("prof-bio").value.trim(),
              avatar_color: document.getElementById("prof-avatar-color").value,
            }),
          });
          toast("Profile updated", "success");
          loadProfile();
        } catch (err) {
          toast(err.message, "error");
        }
      });
    } catch (err) {
      userContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load profile: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Settings
  // ---------------------------------------------------------------------------
  async function loadSettings() {
    try {
      const { settings } = await api("user/settings");
      userContent.innerHTML = `
        <div class="panel-card">
          <h3>Change Password</h3>
          <div class="form-group">
            <label>Current Password</label>
            <input type="password" id="set-current-password" placeholder="Current password">
          </div>
          <div class="form-group">
            <label>New Password</label>
            <input type="password" id="set-new-password" placeholder="Min 6 characters">
          </div>
          <div class="form-actions">
            <button class="btn btn-primary" id="btn-change-password">Change Password</button>
          </div>
        </div>
        <div class="panel-card">
          <h3>Appearance</h3>
          <div class="form-group">
            <label>Theme</label>
            <select id="set-theme">
              <option value="dark" ${document.documentElement.getAttribute("data-theme") === "dark" ? "selected" : ""}>Dark</option>
              <option value="light" ${document.documentElement.getAttribute("data-theme") === "light" ? "selected" : ""}>Light</option>
            </select>
          </div>
          <div class="form-actions">
            <button class="btn btn-primary" id="btn-save-theme">Save Theme</button>
          </div>
        </div>
      `;

      document.getElementById("btn-change-password").addEventListener("click", async () => {
        const currentPassword = document.getElementById("set-current-password").value;
        const newPassword = document.getElementById("set-new-password").value;
        if (!currentPassword || !newPassword) {
          return toast("All password fields are required", "error");
        }
        if (newPassword.length < 6) {
          return toast("New password must be at least 6 characters", "error");
        }
        try {
          await api("user/change-password", {
            method: "POST",
            body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
          });
          toast("Password changed successfully", "success");
          document.getElementById("set-current-password").value = "";
          document.getElementById("set-new-password").value = "";
        } catch (err) {
          toast(err.message, "error");
        }
      });

      document.getElementById("btn-save-theme").addEventListener("click", () => {
        const theme = document.getElementById("set-theme").value;
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("netcradus_theme", theme);
        updateThemeIcon(theme);
        toast("Theme saved", "success");
      });
    } catch (err) {
      userContent.innerHTML = `<div class="panel-card"><p style="color:var(--accent-red)">Failed to load settings: ${escapeHtml(err.message)}</p></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Polling helpers
  // ---------------------------------------------------------------------------
  function stopPolling() {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
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
    initTheme();
    if (token) {
      api("user/auth/me")
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

  function formatDate(ts) {
    if (!ts) return "—";
    return new Date(ts * 1000).toLocaleDateString();
  }
})();