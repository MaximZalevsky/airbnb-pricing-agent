/* chat.js — Airbnb Pricing Agent frontend */

'use strict';

// ─── Constants ───────────────────────────────────────────────────────────────

const SESSION_KEY = 'airbnb_session_id';
const API_URL     = '/api/chat';

const TIER_LABELS = {
  he: {
    budget:       'תקציבי',
    'mid-range':  'ביניים',
    upscale:      'יוקרתי',
    luxury:       'לוקסוס',
    unknown:      'לא ידוע',
  },
  en: {
    budget:       'Budget',
    'mid-range':  'Mid-Range',
    upscale:      'Upscale',
    luxury:       'Luxury',
    unknown:      'Unknown',
  },
};

const TIER_CSS_CLASS = {
  'budget':    'tier-budget',
  'mid-range': 'tier-midrange',
  'upscale':   'tier-upscale',
  'luxury':    'tier-luxury',
  'unknown':   'tier-unknown',
};

// ─── State ────────────────────────────────────────────────────────────────────

let sessionId = localStorage.getItem(SESSION_KEY) || null;

// ─── DOM References ───────────────────────────────────────────────────────────

const chatWindow       = document.getElementById('chat-window');
const userInput        = document.getElementById('user-input');
const sendBtn          = document.getElementById('send-btn');
const typingIndicator  = document.getElementById('typing-indicator');
const newChatBtn       = document.getElementById('new-chat-btn');

// ─── Core: Send Message ───────────────────────────────────────────────────────

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  appendBubble('user', text);
  clearInput();
  showTyping();

  try {
    const res = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text, session_id: sessionId }),
    });

    const data = await res.json();
    hideTyping();

    if (res.status === 400) {
      appendBubble('agent', 'שגיאה בקלט. נא לנסות שוב.');
      return;
    }

    // Persist session
    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);
    }

    // Agent reply
    const replyText = data.reply || '';
    appendBubble('agent', replyText);

    // Recommendation card (only when present)
    if (data.recommendation) {
      appendRecCard(data.recommendation, replyText);
    }

  } catch (err) {
    hideTyping();
    appendBubble('agent', 'שגיאה בחיבור לשרת. נא לבדוק את החיבור ולנסות שוב.');
  }

  scrollToBottom();
}

// ─── Bubble Rendering ────────────────────────────────────────────────────────

function appendBubble(role, text) {
  const div  = document.createElement('div');
  div.className = `bubble bubble-${role}`;
  div.dir       = 'auto';
  div.textContent = text;
  // Insert before the typing indicator so it stays at the bottom of the list
  chatWindow.insertBefore(div, typingIndicator);
  scrollToBottom();
  return div;
}

// ─── Typing Indicator ────────────────────────────────────────────────────────

function showTyping() {
  typingIndicator.classList.remove('hidden');
  scrollToBottom();
}

function hideTyping() {
  typingIndicator.classList.add('hidden');
}

// ─── Input Helpers ────────────────────────────────────────────────────────────

function clearInput() {
  userInput.value = '';
  userInput.style.height = 'auto';
  sendBtn.disabled = true;
}

// ─── Recommendation Card ─────────────────────────────────────────────────────

function appendRecCard(rec, replyText) {
  const lang     = detectLang(replyText);
  const currency = detectCurrency(rec, replyText);
  const isHe     = lang === 'he';

  const card = document.createElement('div');
  card.className = 'rec-card';

  // Header
  const header = el('div', 'rec-header',
    isHe ? '💡 המלצת מחיר' : '💡 Price Recommendation'
  );

  // Body
  const body = document.createElement('div');
  body.className = 'rec-body';

  // Price
  const priceDiv = document.createElement('div');
  priceDiv.className = 'rec-price';
  priceDiv.innerHTML =
    `${esc(currency)}${Math.round(rec.recommended_price)}` +
    `<span class="rec-per-night">${isHe ? '/ לילה' : '/ night'}</span>`;

  // Range
  const rangeDiv = el('div', 'rec-range',
    `${isHe ? 'טווח' : 'Range'}: ` +
    `${esc(currency)}${Math.round(rec.price_min)} – ` +
    `${esc(currency)}${Math.round(rec.price_max)}`
  );

  // Meta: tier badge
  const metaDiv      = document.createElement('div');
  metaDiv.className  = 'rec-meta';
  const tierLabel    = (TIER_LABELS[lang] || TIER_LABELS.he)[rec.tier_label] || rec.tier_label;
  const tierClass    = TIER_CSS_CLASS[rec.tier_label] || 'tier-unknown';
  const tierBadge    = el('span', `tier-badge ${tierClass}`, tierLabel);
  metaDiv.append(tierBadge);

  body.append(priceDiv, rangeDiv, metaDiv);
  card.append(header, body);

  // Cross-city notice
  if (rec.is_cross_city) {
    card.append(buildCrossCityNotice(rec.source_cities, isHe));
  }

  chatWindow.insertBefore(card, typingIndicator);
  scrollToBottom();
}

function buildConfBar(pct, isHe) {
  const wrap     = document.createElement('div');
  wrap.className = 'conf-wrap';

  const container = document.createElement('div');
  container.className = 'conf-bar-container';

  const fill = document.createElement('div');
  fill.className = 'conf-bar-fill';
  fill.style.width = `${pct}%`;

  const label = el('span', 'conf-label',
    `${pct}% ${isHe ? 'ביטחון' : 'Confidence'}`
  );

  container.append(fill);
  wrap.append(container, label);
  return wrap;
}

function buildCrossCityNotice(sourceCities, isHe) {
  const cities = (sourceCities || [])
    .map(c => c.charAt(0).toUpperCase() + c.slice(1))
    .join(', ');

  const notice = document.createElement('div');
  notice.className = 'cross-city-notice';

  const icon = el('span', 'cross-city-icon', '⚠️');
  const text = document.createElement('span');
  text.dir = 'auto';
  text.textContent = isHe
    ? `אין לנו נתוני שוק ישירים לעיר זו. ההמלצה מבוססת על נכסים דומים מ-${cities}.`
    : `We don't have direct data for this city. Recommendation based on similar listings from ${cities}.`;

  notice.append(icon, text);
  return notice;
}

function buildComparables(comparables, currency, isHe) {
  const section = document.createElement('div');
  section.className = 'comparables-section';

  const label = el('div', 'comparables-label',
    isHe ? 'נכסים דומים' : 'Comparable Listings'
  );
  const grid = document.createElement('div');
  grid.className = 'comparables-grid';

  comparables.forEach(comp => {
    grid.append(buildCompCard(comp, currency));
  });

  section.append(label, grid);
  return section;
}

function buildCompCard(comp, currency) {
  const card = document.createElement('div');
  card.className = 'comp-card';

  const name = el('div', 'comp-name', comp.name || 'Unnamed');
  const hood = el('div', 'comp-detail', comp.neighbourhood || '');
  const type = el('div', 'comp-detail', comp.room_type || '');
  const accs = el('div', 'comp-detail',
    `${comp.accommodates || '?'} ${comp.accommodates === 1 ? 'guest' : 'guests'}`
  );
  const price = document.createElement('div');
  price.className = 'comp-price';
  price.innerHTML =
    `${esc(currency)}${Math.round(comp.price)}` +
    `<span class="comp-per-night">/night</span>`;

  card.append(name, hood, type, accs, price);

  if (comp.source_city) {
    const badge = el('span', 'source-city-badge',
      comp.source_city.charAt(0).toUpperCase() + comp.source_city.slice(1)
    );
    card.append(badge);
  }

  return card;
}

// ─── New Chat ─────────────────────────────────────────────────────────────────

function startNewChat() {
  sessionId = null;
  localStorage.removeItem(SESSION_KEY);

  // Remove all bubbles and cards except the typing indicator
  Array.from(chatWindow.children).forEach(child => {
    if (child.id !== 'typing-indicator') child.remove();
  });

  clearInput();
  showWelcome();
  userInput.focus();
}

// ─── Welcome Message ─────────────────────────────────────────────────────────

function showWelcome() {
  appendBubble(
    'agent',
    'שלום! 👋 אני יועץ תמחור Airbnb חכם.\n' +
    'אפשר לתאר לי את הנכס בצורה חופשית — איפה הוא נמצא, איזה סוג נכס זה, כמה אורחים הוא מתאים לארח, או כל פרט אחר שכבר ידוע לך. אני אשאל רק על הפרטים שחסרים כדי לתת המלצת מחיר מדויקת יותר.'
  );
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function detectLang(text) {
  return /[֐-׿]/.test(text) ? 'he' : 'en';
}

function detectCurrency(rec, replyText) {
  return '$';
}

/** Create a simple text element */
function el(tag, className, text) {
  const e = document.createElement(tag);
  e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** Escape a value for safe insertion as text (used inside innerHTML price strings) */
function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ─── Event Listeners ─────────────────────────────────────────────────────────

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

userInput.addEventListener('input', () => {
  // Auto-grow textarea up to 120px
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
  // Enable send only when there is non-whitespace content
  sendBtn.disabled = !userInput.value.trim();
});

newChatBtn.addEventListener('click', startNewChat);

// ─── Init ─────────────────────────────────────────────────────────────────────

showWelcome();
userInput.focus();
