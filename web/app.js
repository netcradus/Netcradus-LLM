/**
 * Netcradus LLM - Professional ChatGPT & Gemini Web Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const appContainer = document.getElementById('app-container');
  const chatInput = document.getElementById('chat-input');
  const btnSend = document.getElementById('btn-send');
  const btnStop = document.getElementById('btn-stop');
  const messagesWrapper = document.getElementById('messages-wrapper');
  const messagesContainer = document.getElementById('messages-container');
  const heroSection = document.getElementById('hero-section');
  const bottomInputContainer = document.getElementById('bottom-input-container');
  const btnNewChat = document.getElementById('btn-new-chat');
  const btnClearChat = document.getElementById('btn-clear-chat');
  const historyList = document.getElementById('history-list');
  const modelSelector = document.getElementById('model-selector');
  const modalModelSelect = document.getElementById('modal-model-select');
  const btnThemeToggle = document.getElementById('btn-theme-toggle');
  const themeIcon = document.getElementById('theme-icon');
  const themeText = document.getElementById('theme-text');

  // Hero Card Input Elements
  const heroChatInput = document.getElementById('hero-chat-input');
  const btnHeroSend = document.getElementById('btn-hero-send');
  const btnHeroAttach = document.getElementById('btn-hero-attach');
  const heroCategorySelect = document.getElementById('hero-category-select');

  // Sidebar Toggles & Input Tools
  const btnSidebarCollapse = document.getElementById('btn-sidebar-collapse');
  const btnSidebarExpand = document.getElementById('btn-sidebar-expand');
  const btnAttach = document.getElementById('btn-attach');
  const btnMic = document.getElementById('btn-mic');
  const fileUploadInput = document.getElementById('file-upload-input');

  // Profile Options & Clear History Elements
  const userProfileTrigger = document.getElementById('user-profile-trigger');
  const headerAvatarTrigger = document.getElementById('header-avatar-trigger');
  const btnSettings = document.getElementById('btn-settings');
  const profileMenuDropdown = document.getElementById('profile-menu-dropdown');
  const btnMenuSettings = document.getElementById('btn-menu-settings');
  const btnMenuClearHistory = document.getElementById('btn-menu-clear-history');
  const btnSidebarClearHistory = document.getElementById('btn-sidebar-clear-history');
  const btnModalClearHistory = document.getElementById('btn-modal-clear-history');
  const settingsModalBackdrop = document.getElementById('settings-modal-backdrop');
  const btnCloseModal = document.getElementById('btn-close-modal');

  // Authentication DOM Elements
  const authOverlay = document.getElementById('auth-overlay');
  const tabSignin = document.getElementById('tab-signin');
  const tabSignup = document.getElementById('tab-signup');
  const formSignin = document.getElementById('form-signin');
  const formSignup = document.getElementById('form-signup');
  const signinEmail = document.getElementById('signin-email');
  const signinPassword = document.getElementById('signin-password');
  const signupName = document.getElementById('signup-name');
  const signupEmail = document.getElementById('signup-email');
  const signupPassword = document.getElementById('signup-password');
  const btnSubmitSignin = document.getElementById('btn-submit-signin');
  const btnSubmitSignup = document.getElementById('btn-submit-signup');
  const signinSpinner = document.getElementById('signin-spinner');
  const signupSpinner = document.getElementById('signup-spinner');
  const btnGoogleSignin = document.getElementById('btn-google-signin');
  const btnGuestMode = document.getElementById('btn-guest-mode');
  const authAlert = document.getElementById('auth-alert');
  const btnForgotPassword = document.getElementById('btn-forgot-password');
  const resetPasswordBackdrop = document.getElementById('reset-password-backdrop');
  const btnCloseResetModal = document.getElementById('btn-close-reset-modal');
  const resetEmail = document.getElementById('reset-email');
  const btnSendResetEmail = document.getElementById('btn-send-reset-email');
  const resetModalAlert = document.getElementById('reset-modal-alert');
  const btnMenuLogout = document.getElementById('btn-menu-logout');
  const btnModalLogout = document.getElementById('btn-modal-logout');

  // User Profile DOM Elements across UI
  const sidebarUserAvatar = document.getElementById('sidebar-user-avatar');
  const sidebarUserName = document.getElementById('sidebar-user-name');
  const sidebarUserPlan = document.getElementById('sidebar-user-plan');
  const dropdownUserName = document.getElementById('dropdown-user-name');
  const dropdownUserBadge = document.getElementById('dropdown-user-badge');
  const dropdownUserEmail = document.getElementById('dropdown-user-email');
  const modalUserAvatar = document.getElementById('modal-user-avatar');
  const modalUserName = document.getElementById('modal-user-name');
  const modalUserEmail = document.getElementById('modal-user-email');
  const modalUserProvider = document.getElementById('modal-user-provider');

  // State Management
  let currentPersona = 'general';
  let selectedModel = 'netcradus-1.0-pro';
  let currentTheme = localStorage.getItem('netcradus_theme') || 'light';
  let isGenerating = false;
  let abortController = null;
  let sessions = loadSessionsFromStorage();
  let currentSessionId = createNewSession();

  // Auth Alert Helper
  function showAuthAlert(elem, msg, type = 'error') {
    if (!elem) return;
    elem.textContent = msg;
    elem.className = `auth-alert ${type}`;
    elem.style.display = 'block';
  }

  function hideAuthAlert(elem) {
    if (!elem) return;
    elem.style.display = 'none';
  }

  // Update Profile UI with logged-in user data
  function updateUserProfileUI(user) {
    if (!user) return;
    const displayName = user.displayName || (user.email ? user.email.split('@')[0] : 'Netcradus User');
    const email = user.email || 'guest@netcradus.local';
    const initial = displayName.charAt(0).toUpperCase() || 'N';
    const providerLabel = user.provider === 'google.com' ? 'Google Auth' :
                         (user.provider === 'firebase-email' ? 'Firebase Auth' : 'Guest Mode');

    if (sidebarUserAvatar) sidebarUserAvatar.textContent = initial;
    if (headerAvatarTrigger) headerAvatarTrigger.textContent = initial;
    if (modalUserAvatar) modalUserAvatar.textContent = initial;
    if (sidebarUserName) sidebarUserName.textContent = displayName;
    if (sidebarUserPlan) sidebarUserPlan.textContent = providerLabel;
    if (dropdownUserName) dropdownUserName.textContent = displayName;
    if (dropdownUserBadge) dropdownUserBadge.textContent = providerLabel;
    if (dropdownUserEmail) dropdownUserEmail.textContent = email;
    if (modalUserName) modalUserName.textContent = displayName;
    if (modalUserEmail) modalUserEmail.textContent = email;
    if (modalUserProvider) modalUserProvider.textContent = `Connected via ${providerLabel}`;
  }

  // Auth Overlay Visibility
  function showAuthOverlay() {
    if (authOverlay) authOverlay.style.display = 'flex';
    if (appContainer) appContainer.style.display = 'none';
  }

  function hideAuthOverlay() {
    if (authOverlay) authOverlay.style.display = 'none';
    if (appContainer) appContainer.style.display = 'flex';
  }

  // Initialize Auth Observer
  if (window.NetcradusAuth) {
    window.NetcradusAuth.onAuthUserStateChanged((user) => {
      if (user) {
        updateUserProfileUI(user);
        hideAuthOverlay();
      } else {
        showAuthOverlay();
      }
    });
  } else {
    hideAuthOverlay();
  }

  // Auth Tab Switching
  if (tabSignin && tabSignup) {
    tabSignin.addEventListener('click', () => {
      tabSignin.classList.add('active');
      tabSignup.classList.remove('active');
      formSignin.style.display = 'flex';
      formSignup.style.display = 'none';
      hideAuthAlert(authAlert);
    });

    tabSignup.addEventListener('click', () => {
      tabSignup.classList.add('active');
      tabSignin.classList.remove('active');
      formSignup.style.display = 'flex';
      formSignin.style.display = 'none';
      hideAuthAlert(authAlert);
    });
  }

  // Sign In Form Submission
  if (formSignin) {
    formSignin.addEventListener('submit', async (e) => {
      e.preventDefault();
      hideAuthAlert(authAlert);

      const email = signinEmail ? signinEmail.value.trim() : '';
      const password = signinPassword ? signinPassword.value : '';

      if (!email || !password) {
        showAuthAlert(authAlert, 'Please fill in all email and password fields.');
        return;
      }

      try {
        if (btnSubmitSignin) btnSubmitSignin.disabled = true;
        if (signinSpinner) signinSpinner.style.display = 'inline-block';
        const user = await window.NetcradusAuth.signInEmailPassword(email, password);
        updateUserProfileUI(user);
        hideAuthOverlay();
      } catch (err) {
        console.error("Sign in failed:", err);
        const errMsg = err.message || 'Failed to sign in. Please check your email and password.';
        showAuthAlert(authAlert, errMsg);
      } finally {
        if (btnSubmitSignin) btnSubmitSignin.disabled = false;
        if (signinSpinner) signinSpinner.style.display = 'none';
      }
    });
  }

  // Create Account Form Submission
  if (formSignup) {
    formSignup.addEventListener('submit', async (e) => {
      e.preventDefault();
      hideAuthAlert(authAlert);

      const name = signupName ? signupName.value.trim() : '';
      const email = signupEmail ? signupEmail.value.trim() : '';
      const password = signupPassword ? signupPassword.value : '';

      if (!email || !password) {
        showAuthAlert(authAlert, 'Please fill in all required fields.');
        return;
      }

      try {
        if (btnSubmitSignup) btnSubmitSignup.disabled = true;
        if (signupSpinner) signupSpinner.style.display = 'inline-block';
        const user = await window.NetcradusAuth.signUpEmailPassword(email, password, name);
        updateUserProfileUI(user);
        hideAuthOverlay();
      } catch (err) {
        console.error("Sign up failed:", err);
        const errMsg = err.message || 'Failed to create account. Email may already be in use.';
        showAuthAlert(authAlert, errMsg);
      } finally {
        if (btnSubmitSignup) btnSubmitSignup.disabled = false;
        if (signupSpinner) signupSpinner.style.display = 'none';
      }
    });
  }

  // Google Sign In Button
  if (btnGoogleSignin) {
    btnGoogleSignin.addEventListener('click', async () => {
      hideAuthAlert(authAlert);
      try {
        const user = await window.NetcradusAuth.signInWithGoogle();
        updateUserProfileUI(user);
        hideAuthOverlay();
      } catch (err) {
        console.error("Google sign in failed:", err);
        showAuthAlert(authAlert, err.message || 'Google sign in canceled or failed.');
      }
    });
  }

  // Guest Mode Button
  if (btnGuestMode) {
    btnGuestMode.addEventListener('click', () => {
      const guest = window.NetcradusAuth.continueAsGuest();
      updateUserProfileUI(guest);
      hideAuthOverlay();
    });
  }

  // Password Reset Modal Controls
  if (btnForgotPassword) {
    btnForgotPassword.addEventListener('click', (e) => {
      e.preventDefault();
      if (resetPasswordBackdrop) resetPasswordBackdrop.style.display = 'flex';
      hideAuthAlert(resetModalAlert);
    });
  }

  if (btnCloseResetModal) {
    btnCloseResetModal.addEventListener('click', () => {
      if (resetPasswordBackdrop) resetPasswordBackdrop.style.display = 'none';
    });
  }

  if (btnSendResetEmail) {
    btnSendResetEmail.addEventListener('click', async () => {
      const email = resetEmail ? resetEmail.value.trim() : '';
      if (!email) {
        showAuthAlert(resetModalAlert, 'Please enter your account email address.');
        return;
      }
      try {
        await window.NetcradusAuth.sendPasswordReset(email);
        showAuthAlert(resetModalAlert, 'Password reset link sent! Check your email inbox.', 'success');
        setTimeout(() => {
          if (resetPasswordBackdrop) resetPasswordBackdrop.style.display = 'none';
        }, 2500);
      } catch (err) {
        showAuthAlert(resetModalAlert, err.message || 'Failed to send reset email.');
      }
    });
  }

  // Log Out Controls
  const handleLogout = async () => {
    if (profileMenuDropdown) profileMenuDropdown.style.display = 'none';
    if (settingsModalBackdrop) settingsModalBackdrop.style.display = 'none';
    await window.NetcradusAuth.signOutUser();
    showAuthOverlay();
  };

  if (btnMenuLogout) btnMenuLogout.addEventListener('click', handleLogout);
  if (btnModalLogout) btnModalLogout.addEventListener('click', handleLogout);

  // Initialize UI
  renderHistoryList();
  applyTheme(currentTheme);

  // Profile Options Dropdown Toggle
  const toggleProfileMenu = (e) => {
    e.stopPropagation();
    if (profileMenuDropdown) {
      const isShown = profileMenuDropdown.style.display === 'block';
      profileMenuDropdown.style.display = isShown ? 'none' : 'block';
    }
  };

  if (userProfileTrigger) userProfileTrigger.addEventListener('click', toggleProfileMenu);
  if (headerAvatarTrigger) headerAvatarTrigger.addEventListener('click', toggleProfileMenu);
  if (btnSettings) btnSettings.addEventListener('click', toggleProfileMenu);

  document.addEventListener('click', (e) => {
    if (profileMenuDropdown && !profileMenuDropdown.contains(e.target) && !e.target.closest('#user-profile-trigger') && !e.target.closest('#header-avatar-trigger')) {
      profileMenuDropdown.style.display = 'none';
    }
  });

  // Settings Modal Controls
  if (btnMenuSettings) {
    btnMenuSettings.addEventListener('click', () => {
      if (profileMenuDropdown) profileMenuDropdown.style.display = 'none';
      if (settingsModalBackdrop) settingsModalBackdrop.style.display = 'flex';
    });
  }

  if (btnCloseModal) {
    btnCloseModal.addEventListener('click', () => {
      if (settingsModalBackdrop) settingsModalBackdrop.style.display = 'none';
    });
  }

  if (settingsModalBackdrop) {
    settingsModalBackdrop.addEventListener('click', (e) => {
      if (e.target === settingsModalBackdrop) {
        settingsModalBackdrop.style.display = 'none';
      }
    });
  }

  // Clear History Actions
  const handleClearHistory = () => {
    clearAllHistory();
  };

  if (btnMenuClearHistory) btnMenuClearHistory.addEventListener('click', handleClearHistory);
  if (btnSidebarClearHistory) btnSidebarClearHistory.addEventListener('click', handleClearHistory);
  if (btnModalClearHistory) btnModalClearHistory.addEventListener('click', handleClearHistory);

  function clearAllHistory() {
    if (confirm('Are you sure you want to clear all chat history?')) {
      sessions = {};
      saveSessionsToStorage();
      startNewSession();
      if (profileMenuDropdown) profileMenuDropdown.style.display = 'none';
      if (settingsModalBackdrop) settingsModalBackdrop.style.display = 'none';
    }
  }

  // Theme Toggle Listener
  if (btnThemeToggle) {
    btnThemeToggle.addEventListener('click', () => {
      currentTheme = currentTheme === 'light' ? 'dark' : 'light';
      applyTheme(currentTheme);
      localStorage.setItem('netcradus_theme', currentTheme);
    });
  }

  function applyTheme(theme) {
    if (theme === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
      if (themeIcon) themeIcon.textContent = '🌙';
      if (themeText) themeText.textContent = 'Dark';
    } else {
      document.documentElement.setAttribute('data-theme', 'dark');
      if (themeIcon) themeIcon.textContent = '☀️';
      if (themeText) themeText.textContent = 'Light';
    }
  }

  // Model Selector
  if (modelSelector) {
    modelSelector.addEventListener('change', (e) => {
      selectedModel = e.target.value;
      if (modalModelSelect) modalModelSelect.value = selectedModel;
    });
  }

  if (modalModelSelect) {
    modalModelSelect.addEventListener('change', (e) => {
      selectedModel = e.target.value;
      if (modelSelector) modelSelector.value = selectedModel;
    });
  }

  // Hero Category Selector
  if (heroCategorySelect) {
    heroCategorySelect.addEventListener('change', (e) => {
      const cat = e.target.value;
      if (cat === 'code') currentPersona = 'code';
      else if (cat === 'cybersecurity' || cat === 'threats') currentPersona = 'reasoning';
      else currentPersona = 'general';
    });
  }

  // Sidebar Collapse / Expand
  if (btnSidebarCollapse && btnSidebarExpand) {
    btnSidebarCollapse.addEventListener('click', () => {
      appContainer.classList.add('sidebar-collapsed');
      btnSidebarExpand.style.display = 'flex';
    });

    btnSidebarExpand.addEventListener('click', () => {
      appContainer.classList.remove('sidebar-collapsed');
      btnSidebarExpand.style.display = 'none';
    });
  }

  // Attachment & Voice Simulator
  const handleAttachClick = () => { if (fileUploadInput) fileUploadInput.click(); };
  if (btnAttach) btnAttach.addEventListener('click', handleAttachClick);
  if (btnHeroAttach) btnHeroAttach.addEventListener('click', handleAttachClick);

  if (fileUploadInput) {
    fileUploadInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        const filename = e.target.files[0].name;
        if (heroChatInput) heroChatInput.value += ` [Attached: ${filename}] `;
        if (chatInput) chatInput.value += ` [Attached: ${filename}] `;
      }
    });
  }

  if (btnMic) {
    btnMic.addEventListener('click', () => {
      btnMic.classList.toggle('active-mic');
      if (btnMic.classList.contains('active-mic')) {
        chatInput.placeholder = "Listening... Speak your prompt...";
      } else {
        chatInput.placeholder = "Ask Netcradus LLM anything...";
      }
    });
  }

  // Textarea Listeners
  if (chatInput) {
    chatInput.addEventListener('input', () => {
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  if (heroChatInput) {
    heroChatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = heroChatInput.value.trim();
        if (text) {
          heroChatInput.value = '';
          sendMessage(text);
        }
      }
    });
  }

  if (btnHeroSend) {
    btnHeroSend.addEventListener('click', () => {
      if (heroChatInput) {
        const text = heroChatInput.value.trim();
        if (text) {
          heroChatInput.value = '';
          sendMessage(text);
        }
      }
    });
  }

  if (btnSend) btnSend.addEventListener('click', () => sendMessage());
  if (btnStop) btnStop.addEventListener('click', stopGeneration);
  if (btnNewChat) btnNewChat.addEventListener('click', () => startNewSession());
  if (btnClearChat) btnClearChat.addEventListener('click', () => clearCurrentSession());

  // Session Storage Helpers
  function loadSessionsFromStorage() {
    try {
      const data = localStorage.getItem('netcradus_sessions');
      return data ? JSON.parse(data) : {};
    } catch {
      return {};
    }
  }

  function saveSessionsToStorage() {
    try {
      localStorage.setItem('netcradus_sessions', JSON.stringify(sessions));
    } catch (e) {
      console.warn('LocalStorage save error:', e);
    }
  }

  function createNewSession() {
    const id = 'session_' + Date.now();
    sessions[id] = {
      id: id,
      title: 'New Conversation',
      messages: [],
      persona: currentPersona,
      createdAt: new Date().toISOString()
    };
    saveSessionsToStorage();
    return id;
  }

  function startNewSession() {
    if (isGenerating) stopGeneration();
    currentSessionId = createNewSession();
    renderCurrentSessionMessages();
    renderHistoryList();
  }

  function clearCurrentSession() {
    if (isGenerating) stopGeneration();
    if (sessions[currentSessionId]) {
      sessions[currentSessionId].messages = [];
      saveSessionsToStorage();
    }
    renderCurrentSessionMessages();
  }

  function renderHistoryList() {
    historyList.innerHTML = '';
    const sorted = Object.values(sessions).sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    sorted.forEach((sess) => {
      const item = document.createElement('div');
      item.className = `history-item ${sess.id === currentSessionId ? 'active' : ''}`;
      item.innerHTML = `
        <svg class="history-item-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span class="history-item-title">${escapeHtml(sess.title)}</span>
        <div class="history-item-actions">
          <button class="icon-btn btn-del" title="Delete Session">🗑</button>
        </div>
      `;

      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('btn-del')) {
          e.stopPropagation();
          deleteSession(sess.id);
        } else {
          switchSession(sess.id);
        }
      });

      historyList.appendChild(item);
    });
  }

  function switchSession(id) {
    if (isGenerating) stopGeneration();
    currentSessionId = id;
    renderCurrentSessionMessages();
    renderHistoryList();
  }

  function deleteSession(id) {
    delete sessions[id];
    saveSessionsToStorage();
    if (currentSessionId === id) {
      const keys = Object.keys(sessions);
      currentSessionId = keys.length > 0 ? keys[0] : createNewSession();
    }
    renderCurrentSessionMessages();
    renderHistoryList();
  }

  function renderCurrentSessionMessages() {
    messagesWrapper.innerHTML = '';
    const sess = sessions[currentSessionId];
    if (!sess || sess.messages.length === 0) {
      messagesWrapper.appendChild(heroSection);
      heroSection.style.display = 'block';
      if (bottomInputContainer) bottomInputContainer.style.display = 'none';
      return;
    }

    heroSection.style.display = 'none';
    if (bottomInputContainer) bottomInputContainer.style.display = 'flex';
    sess.messages.forEach((msg) => {
      appendMessageRowUI(msg.role, msg.content, msg.metrics);
    });
    scrollToBottom();
  }

  // Send Message & Stream Response
  async function sendMessage(overrideText = null) {
    const text = overrideText || (chatInput ? chatInput.value.trim() : '');
    if (!text || isGenerating) return;

    const sess = sessions[currentSessionId];
    if (!sess) return;

    if (sess.messages.length === 0) {
      sess.title = text.length > 28 ? text.substring(0, 28) + '...' : text;
    }

    heroSection.style.display = 'none';
    if (bottomInputContainer) bottomInputContainer.style.display = 'flex';

    // Append User Message
    sess.messages.push({ role: 'user', content: text });
    appendMessageRowUI('user', text);
    saveSessionsToStorage();
    renderHistoryList();

    // Reset Input
    if (chatInput) {
      chatInput.value = '';
      chatInput.style.height = 'auto';
    }

    // Prepare AI UI Row
    const aiRow = appendMessageRowUI('assistant', '');
    const bubble = aiRow.querySelector('.message-bubble');
    const footer = aiRow.querySelector('.message-footer');
    bubble.innerHTML = `<div class="typing-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;

    isGenerating = true;
    if (btnSend) btnSend.style.display = 'none';
    if (btnStop) btnStop.style.display = 'flex';
    abortController = new AbortController();

    let fullText = '';

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortController.signal,
        body: JSON.stringify({
          messages: sess.messages,
          persona: currentPersona,
          model: selectedModel
        })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').trim();
            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);
              if (data.chunk) {
                fullText += data.chunk;
                bubble.innerHTML = formatMarkdown(fullText);
                attachCopyCodeListeners(bubble);
                scrollToBottom();
              }
              if (data.done && data.metrics) {
                const m = data.metrics;
                footer.innerHTML = `
                  <div class="message-actions-left">
                    <button class="btn-action btn-copy-msg" title="Copy Response">📋 Copy</button>
                    <button class="btn-action btn-thumb btn-thumb-up" title="Good Response">👍</button>
                    <button class="btn-action btn-thumb btn-thumb-down" title="Bad Response">👎</button>
                    <button class="btn-action btn-regen" title="Regenerate">🔄 Regenerate</button>
                  </div>
                  <span class="time-badge">${m.time_sec}s</span>
                `;
                attachMessageActions(aiRow, fullText, text);
              }
            } catch (e) {
              console.warn('JSON stream parse error:', e);
            }
          }
        }
      }

      // Save to session history
      sess.messages.push({ role: 'assistant', content: fullText });
      saveSessionsToStorage();

    } catch (err) {
      if (err.name === 'AbortError') {
        bubble.innerHTML += `<br><em style="color: var(--text-dim); font-size: 0.85rem;">[Generation Stopped]</em>`;
      } else {
        bubble.innerHTML = `<span style="color: #ef4444;">⚠️ Error communicating with Netcradus LLM: ${err.message}</span>`;
      }
    } finally {
      isGenerating = false;
      if (btnSend) btnSend.style.display = 'flex';
      if (btnStop) btnStop.style.display = 'none';
      abortController = null;
    }
  }

  function stopGeneration() {
    if (abortController) {
      abortController.abort();
    }
  }

  // Append Message Row UI
  function appendMessageRowUI(role, content, metrics = null) {
    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = `avatar ${role === 'user' ? 'user-avatar' : 'ai-avatar'}`;
    avatar.textContent = role === 'user' ? 'N' : 'AI';

    const wrapper = document.createElement('div');
    wrapper.className = 'message-bubble-wrapper';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = content ? formatMarkdown(content) : '';
    attachCopyCodeListeners(bubble);

    const footer = document.createElement('div');
    footer.className = 'message-footer';

    if (role === 'assistant' && content) {
      const durationText = metrics && metrics.time_sec ? `${metrics.time_sec}s` : '';
      footer.innerHTML = `
        <div class="message-actions-left">
          <button class="btn-action btn-copy-msg" title="Copy Response">📋 Copy</button>
          <button class="btn-action btn-thumb btn-thumb-up" title="Good Response">👍</button>
          <button class="btn-action btn-thumb btn-thumb-down" title="Bad Response">👎</button>
          <button class="btn-action btn-regen" title="Regenerate">🔄 Regenerate</button>
        </div>
        ${durationText ? `<span class="time-badge">${durationText}</span>` : ''}
      `;
      attachMessageActions(row, content, null);
    }

    wrapper.appendChild(bubble);
    wrapper.appendChild(footer);

    row.appendChild(avatar);
    row.appendChild(wrapper);
    messagesWrapper.appendChild(row);

    scrollToBottom();
    return row;
  }

  function attachMessageActions(row, aiContent, userQuery) {
    const copyBtn = row.querySelector('.btn-copy-msg');
    const thumbUp = row.querySelector('.btn-thumb-up');
    const thumbDown = row.querySelector('.btn-thumb-down');
    const regenBtn = row.querySelector('.btn-regen');

    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        const temp = document.createElement('div');
        temp.innerHTML = aiContent;
        const textToCopy = temp.innerText || aiContent;
        navigator.clipboard.writeText(textToCopy).then(() => {
          copyBtn.textContent = '✓ Copied!';
          setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
        });
      });
    }

    if (thumbUp) {
      thumbUp.addEventListener('click', () => {
        thumbUp.classList.toggle('active-feedback');
        if (thumbDown) thumbDown.classList.remove('active-feedback');
      });
    }

    if (thumbDown) {
      thumbDown.addEventListener('click', () => {
        thumbDown.classList.toggle('active-feedback');
        if (thumbUp) thumbUp.classList.remove('active-feedback');
      });
    }

    if (regenBtn) {
      regenBtn.addEventListener('click', () => {
        const sess = sessions[currentSessionId];
        if (sess && sess.messages.length >= 2) {
          sess.messages.pop();
          const lastUser = sess.messages.pop();
          saveSessionsToStorage();
          renderCurrentSessionMessages();
          sendMessage(lastUser.content);
        }
      });
    }
  }

  // Full Markdown Renderer
  function formatMarkdown(text) {
    if (!text) return '';

    let formatted = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = (lang || 'code').toUpperCase();
      return `
        <div class="code-wrapper">
          <div class="code-header">
            <span class="lang-label">${language}</span>
            <button class="btn-copy-code">Copy</button>
          </div>
          <pre><code>${code.trim()}</code></pre>
        </div>
      `;
    });

    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    const parts = formatted.split(/(<div class="code-wrapper">[\s\S]*?<\/div>)/g);
    for (let i = 0; i < parts.length; i++) {
      if (!parts[i].startsWith('<div class="code-wrapper">')) {
        parts[i] = parts[i].replace(/\n/g, '<br>');
      }
    }

    return parts.join('');
  }

  function attachCopyCodeListeners(container) {
    const copyBtns = container.querySelectorAll('.btn-copy-code');
    copyBtns.forEach(btn => {
      btn.onclick = function() {
        const preElem = btn.closest('.code-wrapper').querySelector('code');
        if (preElem) {
          navigator.clipboard.writeText(preElem.innerText).then(() => {
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
          });
        }
      };
    });
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
});
