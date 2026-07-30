/**
 * Netcradus LLM - Professional ChatGPT & Gemini Web Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const chatInput = document.getElementById('chat-input');
  const btnSend = document.getElementById('btn-send');
  const btnStop = document.getElementById('btn-stop');
  const messagesWrapper = document.getElementById('messages-wrapper');
  const messagesContainer = document.getElementById('messages-container');
  const heroSection = document.getElementById('hero-section');
  const btnNewChat = document.getElementById('btn-new-chat');
  const btnClearChat = document.getElementById('btn-clear-chat');
  const historyList = document.getElementById('history-list');
  const suggestionCards = document.querySelectorAll('.suggestion-card');
  const personaBtns = document.querySelectorAll('.persona-btn');
  const activePersonaBadge = document.getElementById('active-persona-badge');

  // Sliders
  const tempSlider = document.getElementById('temp-slider');
  const tempVal = document.getElementById('temp-val');
  const tokensSlider = document.getElementById('tokens-slider');
  const tokensVal = document.getElementById('tokens-val');

  // State Management
  let currentPersona = 'general';
  let isGenerating = false;
  let abortController = null;
  let sessions = loadSessionsFromStorage();
  let currentSessionId = createNewSession();

  // Initialize UI
  renderHistoryList();
  updatePersonaBadge();

  // Sliders
  tempSlider.addEventListener('input', (e) => {
    tempVal.textContent = parseFloat(e.target.value).toFixed(1);
  });
  tokensSlider.addEventListener('input', (e) => {
    tokensVal.textContent = e.target.value;
  });

  // Persona Buttons
  personaBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      personaBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentPersona = btn.getAttribute('data-persona') || 'general';
      updatePersonaBadge();
    });
  });

  function updatePersonaBadge() {
    const names = {
      general: 'General Mode',
      code: 'Coding Expert Mode',
      reasoning: 'Deep Reasoning Mode',
      creative: 'Creative Mode'
    };
    activePersonaBadge.textContent = names[currentPersona] || 'General Mode';
  }

  // Textarea Auto-expand
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

  btnSend.addEventListener('click', () => sendMessage());
  btnStop.addEventListener('click', stopGeneration);
  btnNewChat.addEventListener('click', () => startNewSession());
  btnClearChat.addEventListener('click', () => clearCurrentSession());

  suggestionCards.forEach((card) => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      if (prompt) {
        chatInput.value = prompt;
        sendMessage();
      }
    });
  });

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
      return;
    }

    heroSection.style.display = 'none';
    sess.messages.forEach((msg) => {
      appendMessageRowUI(msg.role, msg.content, msg.metrics);
    });
    scrollToBottom();
  }

  // Send Message & Stream Tokens
  async function sendMessage(overrideText = null) {
    const text = overrideText || chatInput.value.trim();
    if (!text || isGenerating) return;

    const sess = sessions[currentSessionId];
    if (!sess) return;

    if (sess.messages.length === 0) {
      sess.title = text.length > 28 ? text.substring(0, 28) + '...' : text;
    }

    heroSection.style.display = 'none';

    // Append User Message
    sess.messages.push({ role: 'user', content: text });
    appendMessageRowUI('user', text);
    saveSessionsToStorage();
    renderHistoryList();

    // Reset Input
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Prepare AI UI Row
    const aiRow = appendMessageRowUI('assistant', '');
    const bubble = aiRow.querySelector('.message-bubble');
    const footer = aiRow.querySelector('.message-footer');
    bubble.innerHTML = `<div class="typing-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;

    isGenerating = true;
    btnSend.style.display = 'none';
    btnStop.style.display = 'flex';
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
          temperature: parseFloat(tempSlider.value),
          max_new_tokens: parseInt(tokensSlider.value, 10)
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
                  <span class="metrics-badge">⚡ ${m.tok_per_sec} tok/s • ${m.tokens} tokens • ${m.time_sec}s</span>
                  <div class="message-actions">
                    <button class="btn-action btn-regen" title="Regenerate">🔄 Regenerate</button>
                  </div>
                `;
                attachMessageActions(aiRow, text);
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
        bubble.innerHTML += `<br><em style="color: #94a3b8; font-size: 0.85rem;">[Generation Stopped]</em>`;
      } else {
        bubble.innerHTML = `<span style="color: #ef4444;">⚠️ Error communicating with Netcradus LLM: ${err.message}</span>`;
      }
    } finally {
      isGenerating = false;
      btnSend.style.display = 'flex';
      btnStop.style.display = 'none';
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
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const wrapper = document.createElement('div');
    wrapper.className = 'message-bubble-wrapper';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = content ? formatMarkdown(content) : '';
    attachCopyCodeListeners(bubble);

    const footer = document.createElement('div');
    footer.className = 'message-footer';

    if (role === 'assistant' && metrics) {
      footer.innerHTML = `
        <span class="metrics-badge">⚡ ${metrics.tok_per_sec} tok/s • ${metrics.tokens} tokens</span>
        <div class="message-actions">
          <button class="btn-action btn-regen">🔄 Regenerate</button>
        </div>
      `;
      attachMessageActions(row, content);
    }

    wrapper.appendChild(bubble);
    wrapper.appendChild(footer);

    row.appendChild(avatar);
    row.appendChild(wrapper);
    messagesWrapper.appendChild(row);

    scrollToBottom();
    return row;
  }

  function attachMessageActions(row, userQuery) {
    const regenBtn = row.querySelector('.btn-regen');
    if (regenBtn) {
      regenBtn.addEventListener('click', () => {
        const sess = sessions[currentSessionId];
        if (sess && sess.messages.length >= 2) {
          sess.messages.pop(); // remove last AI
          const lastUser = sess.messages.pop(); // remove last user
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

    // Code Blocks ```lang \n code ```
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

    // Inline code `code`
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold **text**
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic *text*
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Convert linebreaks outside pre blocks
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
