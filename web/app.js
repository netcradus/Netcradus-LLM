/**
 * Netcradus LLM - Production Application Controller (app.js)
 * 
 * Features:
 * - Real-time SSE Token Streaming with multi-turn session persistence.
 * - Persona switching (General, Coding Expert, Deep Reasoning, Creative Security).
 * - Hero input card & quick-start capability prompt cards.
 * - Speech-to-Text Voice Dictation (Web Speech API).
 * - File Attachment reader (code, text, log files).
 * - Markdown & Collapsible Thought Process (<details class='reasoning-block'>) parser.
 * - Light & Dark Theme Switcher with localStorage persistence.
 */
import { auth } from "./firebase-config.js";
document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Main Layout & Navigation
  const appContainer = document.getElementById('app-container');
  const sidebar = document.getElementById('sidebar');
  const btnSidebarCollapse = document.getElementById('btn-sidebar-collapse');
  const btnSidebarExpand = document.getElementById('btn-sidebar-expand');
  const btnNewChat = document.getElementById('btn-new-chat');
  const btnClearChat = document.getElementById('btn-clear-chat');
  const historyList = document.getElementById('history-list');
  const btnSidebarClearHistory = document.getElementById('btn-sidebar-clear-history');

  // DOM Elements - Hero Section & Input Tools
  const heroSection = document.getElementById('hero-section');
  const heroChatInput = document.getElementById('hero-chat-input');
  const btnHeroSend = document.getElementById('btn-hero-send');
  const btnHeroAttach = document.getElementById('btn-hero-attach');
  const heroAttachmentPreview = document.getElementById('hero-attachment-preview');
  const quickCards = document.querySelectorAll('.quick-card');

  // DOM Elements - Bottom Chat Input Bar
  const bottomInputContainer = document.getElementById('bottom-input-container');
  const bottomAttachmentPreview = document.getElementById('bottom-attachment-preview');
  const chatInput = document.getElementById('chat-input');
  const btnSend = document.getElementById('btn-send');
  const btnStop = document.getElementById('btn-stop');
  const btnAttach = document.getElementById('btn-attach');
  const btnMic = document.getElementById('btn-mic');
  const fileUploadInput = document.getElementById('file-upload-input');
  const messagesContainer = document.getElementById('messages-container');
  const messagesWrapper = document.getElementById('messages-wrapper');

  // DOM Elements - Header, Persona & Theme
  const modelSelector = document.getElementById('model-selector');
  const modalModelSelect = document.getElementById('modal-model-select');
  const btnThemeToggle = document.getElementById('btn-theme-toggle');
  const btnModalThemeLight = document.getElementById('btn-modal-theme-light');
  const btnModalThemeDark = document.getElementById('btn-modal-theme-dark');
  const headerPersonaPills = document.querySelectorAll('#header-persona-pills .persona-pill');
  const heroPersonaBtns = document.querySelectorAll('.hero-persona-btn');

  // DOM Elements - User Profile & Modals
  const userProfileTrigger = document.getElementById('user-profile-trigger');
  const headerAvatarTrigger = document.getElementById('header-avatar-trigger');
  const btnSettings = document.getElementById('btn-settings');
  const profileMenuDropdown = document.getElementById('profile-menu-dropdown');
  const btnMenuSettings = document.getElementById('btn-menu-settings');
  const btnMenuClearHistory = document.getElementById('btn-menu-clear-history');
  const btnMenuLogout = document.getElementById('btn-menu-logout');
  const btnModalLogout = document.getElementById('btn-modal-logout');
  const settingsModalBackdrop = document.getElementById('settings-modal-backdrop');
  const btnCloseModal = document.getElementById('btn-close-modal');

  // User Profile Labels
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

  // Application State
  let currentPersona = 'general';
  let selectedModel = 'netcradus-1.0-pro';
  let currentTheme = localStorage.getItem('netcradus_theme') || 'light';
  let isGenerating = false;
  let abortController = null;
  let attachedFile = null; // { name, content, size }
  let speechRecognition = null;
  let isRecording = false;

  let sessions = loadSessionsFromStorage();
  let currentSessionId = createNewSession();

  // Initialize UI & Theme
  applyTheme(currentTheme);
  initDefaultUserProfileUI();
  renderHistoryList();
  renderCurrentSessionMessages();

  // ---------------------------------------------------------------------------
  // Theme Toggle Management
  // ---------------------------------------------------------------------------
  function applyTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('netcradus_theme', theme);
    if (btnThemeToggle) {
      btnThemeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
  }

  function toggleTheme() {
    applyTheme(currentTheme === 'light' ? 'dark' : 'light');
  }

  if (btnThemeToggle) btnThemeToggle.addEventListener('click', toggleTheme);
  if (btnModalThemeLight) btnModalThemeLight.addEventListener('click', () => applyTheme('light'));
  if (btnModalThemeDark) btnModalThemeDark.addEventListener('click', () => applyTheme('dark'));

  // ---------------------------------------------------------------------------
  // Persona Pills & Quick Cards Handler
  // ---------------------------------------------------------------------------
  function setPersona(persona) {
    currentPersona = persona;

    if (headerPersonaPills) {
      headerPersonaPills.forEach(pill => {
        pill.classList.toggle('active', pill.getAttribute('data-persona') === persona);
      });
    }

    if (heroPersonaBtns) {
      heroPersonaBtns.forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-persona') === persona);
      });
    }
  }

  if (headerPersonaPills) {
    headerPersonaPills.forEach(pill => {
      pill.addEventListener('click', () => {
        setPersona(pill.getAttribute('data-persona'));
      });
    });
  }

  if (heroPersonaBtns) {
    heroPersonaBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        setPersona(btn.getAttribute('data-persona'));
      });
    });
  }

  if (quickCards) {
    quickCards.forEach(card => {
      card.addEventListener('click', () => {
        const prompt = card.getAttribute('data-prompt');
        const persona = card.getAttribute('data-persona') || 'general';
        setPersona(persona);
        if (prompt) sendMessage(prompt);
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Model Selector Synchronization
  // ---------------------------------------------------------------------------
  if (modelSelector && modalModelSelect) {
    modelSelector.addEventListener('change', (e) => {
      selectedModel = e.target.value;
      modalModelSelect.value = selectedModel;
    });

    modalModelSelect.addEventListener('change', (e) => {
      selectedModel = e.target.value;
      modelSelector.value = selectedModel;
    });
  }

  // ---------------------------------------------------------------------------
  // File Attachment Handler
  // ---------------------------------------------------------------------------
  function triggerFileUpload() {
    if (fileUploadInput) fileUploadInput.click();
  }

  if (btnAttach) btnAttach.addEventListener('click', triggerFileUpload);
  if (btnHeroAttach) btnHeroAttach.addEventListener('click', triggerFileUpload);

  if (fileUploadInput) {
    fileUploadInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (evt) => {
        attachedFile = {
          name: file.name,
          size: file.size,
          content: evt.target.result
        };
        renderAttachmentPill();
      };
      reader.readAsText(file);
    });
  }

  function renderAttachmentPill() {
    if (!attachedFile) {
      if (heroAttachmentPreview) heroAttachmentPreview.style.display = 'none';
      if (bottomAttachmentPreview) bottomAttachmentPreview.style.display = 'none';
      return;
    }

    const kbSize = (attachedFile.size / 1024).toFixed(1);
    const html = `
      <div class="file-attachment-pill">
        <span>📄 ${escapeHtml(attachedFile.name)} (${kbSize} KB)</span>
        <span class="remove-file" title="Remove File">✕</span>
      </div>
    `;

    if (heroAttachmentPreview) {
      heroAttachmentPreview.innerHTML = html;
      heroAttachmentPreview.style.display = 'block';
      const removeBtn = heroAttachmentPreview.querySelector('.remove-file');
      if (removeBtn) removeBtn.onclick = clearAttachment;
    }

    if (bottomAttachmentPreview) {
      bottomAttachmentPreview.innerHTML = html;
      bottomAttachmentPreview.style.display = 'block';
      const removeBtn = bottomAttachmentPreview.querySelector('.remove-file');
      if (removeBtn) removeBtn.onclick = clearAttachment;
    }
  }

  function clearAttachment() {
    attachedFile = null;
    if (fileUploadInput) fileUploadInput.value = '';
    renderAttachmentPill();
  }

  // ---------------------------------------------------------------------------
  // Voice Input (Speech Recognition)
  // ---------------------------------------------------------------------------
  if (btnMic) {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
      speechRecognition = new SpeechRec();
      speechRecognition.continuous = false;
      speechRecognition.interimResults = false;
      speechRecognition.lang = 'en-US';

      speechRecognition.onstart = () => {
        isRecording = true;
        btnMic.classList.add('recording');
        btnMic.title = 'Listening... Speak now';
      };

      speechRecognition.onresult = (evt) => {
        const transcript = evt.results[0][0].transcript;
        const activeInput = heroSection && heroSection.style.display !== 'none' ? heroChatInput : chatInput;
        if (activeInput) {
          activeInput.value = activeInput.value ? `${activeInput.value} ${transcript}` : transcript;
          activeInput.focus();
        }
      };

      speechRecognition.onerror = (evt) => {
        console.warn('Speech recognition error:', evt.error);
        stopRecording();
      };

      speechRecognition.onend = () => {
        stopRecording();
      };

      btnMic.addEventListener('click', () => {
        if (isRecording) {
          speechRecognition.stop();
        } else {
          speechRecognition.start();
        }
      });
    } else {
      btnMic.title = 'Voice input not supported in this browser';
    }
  }

  function stopRecording() {
    isRecording = false;
    if (btnMic) {
      btnMic.classList.remove('recording');
      btnMic.title = 'Voice Input';
    }
  }

  // ---------------------------------------------------------------------------
  // Session & Local Storage Management
  // ---------------------------------------------------------------------------
  function loadSessionsFromStorage() {
    try {
      const saved = localStorage.getItem('netcradus_sessions');
      return saved ? JSON.parse(saved) : {};
    } catch (e) {
      return {};
    }
  }

  function saveSessionsToStorage() {
    localStorage.setItem('netcradus_sessions', JSON.stringify(sessions));
  }

  function createNewSession() {
    const id = 'sess_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
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
    clearAttachment();
    renderCurrentSessionMessages();
    renderHistoryList();
  }

  function clearCurrentSession() {
    if (isGenerating) stopGeneration();
    if (sessions[currentSessionId]) {
      sessions[currentSessionId].messages = [];
      saveSessionsToStorage();
    }
    clearAttachment();
    renderCurrentSessionMessages();
  }

  function clearAllHistory() {
    if (confirm('Are you sure you want to clear all chat history? This action cannot be undone.')) {
      if (isGenerating) stopGeneration();
      sessions = {};
      localStorage.removeItem('netcradus_sessions');
      currentSessionId = createNewSession();
      clearAttachment();
      renderCurrentSessionMessages();
      renderHistoryList();
      if (profileMenuDropdown) profileMenuDropdown.style.display = 'none';
    }
  }

  function renderHistoryList() {
    if (!historyList) return;
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
    clearAttachment();
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
    if (!messagesWrapper) return;
    messagesWrapper.innerHTML = '';
    const sess = sessions[currentSessionId];

    if (!sess || sess.messages.length === 0) {
      if (heroSection) {
        messagesWrapper.appendChild(heroSection);
        heroSection.style.display = 'flex';
        heroSection.style.flexDirection = 'column';
      }
      if (bottomInputContainer) bottomInputContainer.style.display = 'none';
      return;
    }

    if (heroSection) heroSection.style.display = 'none';
    if (bottomInputContainer) bottomInputContainer.style.display = 'flex';

    sess.messages.forEach((msg) => {
      appendMessageRowUI(msg.role, msg.content, msg.metrics);
    });
    scrollToBottom();
  }

  // ---------------------------------------------------------------------------
  // Send Message & Stream Response
  // ---------------------------------------------------------------------------
  async function sendMessage(overrideText = null) {
    let text = overrideText;
    if (!text) {
      if (heroSection && heroSection.style.display !== 'none' && heroChatInput) {
        text = heroChatInput.value.trim();
      } else if (chatInput) {
        text = chatInput.value.trim();
      }
    }

    if ((!text && !attachedFile) || isGenerating) return;

    // If file is attached, append file content to prompt
    let fullPromptText = text || '';
    if (attachedFile) {
      fullPromptText += `\n\n[Attached File: ${attachedFile.name}]\n\`\`\`\n${attachedFile.content}\n\`\`\``;
    }

    const sess = sessions[currentSessionId];
    if (!sess) return;

    if (sess.messages.length === 0) {
      sess.title = text.length > 28 ? text.substring(0, 28) + '...' : (attachedFile ? attachedFile.name : 'New Chat');
    }

    if (heroSection) heroSection.style.display = 'none';
    if (bottomInputContainer) bottomInputContainer.style.display = 'flex';

    // Append User Message
    sess.messages.push({ role: 'user', content: fullPromptText });
    appendMessageRowUI('user', fullPromptText);
    saveSessionsToStorage();
    renderHistoryList();

    // Reset Inputs & Clear Attachment
    if (heroChatInput) {
      heroChatInput.value = '';
      heroChatInput.style.height = 'auto';
    }
    if (chatInput) {
      chatInput.value = '';
      chatInput.style.height = 'auto';
    }
    clearAttachment();

    // Prepare AI UI Row
    const aiRow = appendMessageRowUI('assistant', '');
    const bubble = aiRow.querySelector('.message-bubble');
    const footer = aiRow.querySelector('.message-footer');
    bubble.innerHTML = `<div class="typing-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;

    isGenerating = true;
    if (btnSend) btnSend.style.display = 'none';
    if (btnStop) btnStop.style.display = 'flex';
    abortController = new AbortController();

    let fullResponseText = '';

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortController.signal,
        body: JSON.stringify({
          messages: sess.messages,
          persona: currentPersona,
          model: selectedModel,
          user: { uid: 'usr_local', name: 'Netcradus User' }
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
                fullResponseText += data.chunk;
                bubble.innerHTML = formatMarkdown(fullResponseText);
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
                  <span class="time-badge">${m.time_sec}s (${m.tok_per_sec} tok/s)</span>
                `;
                attachMessageActions(aiRow, fullResponseText, text);
              }
            } catch (e) {
              console.warn('JSON stream parse error:', e);
            }
          }
        }
      }

      sess.messages.push({ role: 'assistant', content: fullResponseText });
      saveSessionsToStorage();

    } catch (err) {
      if (err.name === 'AbortError') {
        bubble.innerHTML += `<br><em style="color: var(--text-dim); font-size: 0.85rem;">[Generation Stopped]</em>`;
      } else {
        bubble.innerHTML = `<span style="color: #ef4444;">⚠️ Error communicating with Netcradus LLM backend (${err.message}). Please ensure web_server.py is running.</span>`;
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

  // ---------------------------------------------------------------------------
  // Key Listeners for Inputs
  // ---------------------------------------------------------------------------
  if (btnHeroSend) btnHeroSend.addEventListener('click', () => sendMessage());
  if (btnSend) btnSend.addEventListener('click', () => sendMessage());
  if (btnStop) btnStop.addEventListener('click', stopGeneration);

  if (heroChatInput) {
    heroChatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  // Auto-resize chat textarea
  const autoResizeTextarea = (elem) => {
    if (!elem) return;
    elem.style.height = 'auto';
    elem.style.height = Math.min(elem.scrollHeight, 180) + 'px';
  };

  if (heroChatInput) heroChatInput.addEventListener('input', () => autoResizeTextarea(heroChatInput));
  if (chatInput) chatInput.addEventListener('input', () => autoResizeTextarea(chatInput));

  // ---------------------------------------------------------------------------
  // Message UI Row Builder & Action Listeners
  // ---------------------------------------------------------------------------
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
    if (messagesWrapper) messagesWrapper.appendChild(row);

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
          sess.messages.pop(); // Pop assistant msg
          const lastUser = sess.messages.pop(); // Pop user msg
          saveSessionsToStorage();
          renderCurrentSessionMessages();
          sendMessage(lastUser.content);
        }
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Markdown & Reasoning Details Renderer
  // ---------------------------------------------------------------------------
  function formatMarkdown(text) {
    if (!text) return '';

    // Separate collapsible reasoning blocks first to protect their HTML tags
    const reasoningBlocks = [];
    let processed = text.replace(/<details class='reasoning-block'>[\s\S]*?<\/details>/gi, (match) => {
      reasoningBlocks.push(match);
      return `___REASONING_BLOCK_${reasoningBlocks.length - 1}___`;
    });

    // Escape raw unsafe HTML
    processed = processed
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Re-insert protected reasoning HTML blocks
    reasoningBlocks.forEach((block, idx) => {
      processed = processed.replace(`___REASONING_BLOCK_${idx}___`, block);
    });

    // Code Blocks parser
    processed = processed.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
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

    // Inline formatting: code, bold, italics
    processed = processed.replace(/`([^`]+)`/g, '<code>$1</code>');
    processed = processed.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    processed = processed.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Paragraph line breaks outside code blocks and details blocks
    const parts = processed.split(/(<div class="code-wrapper">[\s\S]*?<\/div>|<details class='reasoning-block'>[\s\S]*?<\/details>)/g);
    for (let i = 0; i < parts.length; i++) {
      if (!parts[i].startsWith('<div class="code-wrapper">') && !parts[i].startsWith('<details class=\'reasoning-block\'>')) {
        parts[i] = parts[i].replace(/\n/g, '<br>');
      }
    }

    return parts.join('');
  }

  function attachCopyCodeListeners(container) {
    const copyBtns = container.querySelectorAll('.btn-copy-code');
    copyBtns.forEach(btn => {
      btn.onclick = function () {
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
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ---------------------------------------------------------------------------
  // Sidebar Collapse, New Chat & Modals Handlers
  // ---------------------------------------------------------------------------
  if (btnSidebarCollapse && sidebar && btnSidebarExpand) {
    btnSidebarCollapse.addEventListener('click', () => {
      sidebar.classList.add('collapsed');
      btnSidebarExpand.style.display = 'flex';
    });

    btnSidebarExpand.addEventListener('click', () => {
      sidebar.classList.remove('collapsed');
      btnSidebarExpand.style.display = 'none';
    });
  }

  if (btnNewChat) btnNewChat.addEventListener('click', startNewSession);
  if (btnClearChat) btnClearChat.addEventListener('click', clearCurrentSession);
  if (btnSidebarClearHistory) btnSidebarClearHistory.addEventListener('click', clearAllHistory);
  if (btnMenuClearHistory) btnMenuClearHistory.addEventListener('click', clearAllHistory);


  // Settings Modal Controls
  if (btnSettings && settingsModalBackdrop) {
    btnSettings.addEventListener('click', () => {
      settingsModalBackdrop.style.display = 'flex';
    });
  }

  if (btnMenuSettings && settingsModalBackdrop) {
    btnMenuSettings.addEventListener('click', () => {
      if (profileMenuDropdown) profileMenuDropdown.style.display = 'none';
      settingsModalBackdrop.style.display = 'flex';
    });
  }

  if (btnCloseModal && settingsModalBackdrop) {
    btnCloseModal.addEventListener('click', () => {
      settingsModalBackdrop.style.display = 'none';
    });
  }

  // Profile Popup Dropdown Toggle
  const toggleProfileMenu = (e) => {
    e.stopPropagation();
    if (profileMenuDropdown) {
      const isVisible = profileMenuDropdown.style.display === 'block';
      profileMenuDropdown.style.display = isVisible ? 'none' : 'block';
    }
  };

  if (userProfileTrigger) userProfileTrigger.addEventListener('click', toggleProfileMenu);
  if (headerAvatarTrigger) headerAvatarTrigger.addEventListener('click', toggleProfileMenu);

  document.addEventListener('click', (e) => {
    if (profileMenuDropdown && !profileMenuDropdown.contains(e.target) && e.target !== userProfileTrigger && e.target !== headerAvatarTrigger) {
      profileMenuDropdown.style.display = 'none';
    }
  });

  // Default User Profile UI Setup
  function initDefaultUserProfileUI() {
    const displayName = 'Netcradus User';
    const email = 'user@netcradus.ai';
    const initial = 'N';
    const providerLabel = 'Pro Plan';

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
    if (modalUserProvider) modalUserProvider.textContent = `Status: Active (${providerLabel})`;
  }
});
