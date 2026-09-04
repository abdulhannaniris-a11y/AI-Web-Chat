"""
AI Website Chat Assistant
--------------------------
A single-file FastAPI application that:
  1. Fetches and extracts text content from a user-supplied website URL.
  2. Lets the user ask questions about that content via a Groq-hosted LLM.
  3. Keeps a simple in-memory conversation history tied to the loaded website.

Run with:
    python -m uvicorn main:app --reload
"""

import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load variables from a local .env file (if present) into the environment.
load_dotenv()

# The Groq API key must be supplied via environment variable — never hard-coded.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# The Groq model to use. Defaults to an OpenAI open-weight model hosted on Groq.
# This can be overridden via the GROQ_MODEL environment variable if desired.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Groq's OpenAI-compatible chat completions endpoint.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Maximum number of characters of website text we keep. This keeps the prompt
# well within the model's context window while leaving room for the question,
# system instructions, and conversation history.
MAX_CONTENT_CHARS = 12000

# Maximum number of previous Q&A turns kept in memory per website, to keep the
# prompt size (and therefore token usage) bounded as a conversation grows.
MAX_HISTORY_TURNS = 6

# Maximum tokens the model is allowed to generate per answer.
MAX_ANSWER_TOKENS = 800

# HTTP request timeout (seconds) when fetching a website.
FETCH_TIMEOUT_SECONDS = 10

app = FastAPI(title="AI Website Chat Assistant")

# ---------------------------------------------------------------------------
# In-memory application state
# ---------------------------------------------------------------------------
# This is a single-user prototype: one "current website" and one chat history
# are kept in memory. Loading a new URL replaces both. No database is used
# because the requirements do not call for persistence across restarts or
# multiple simultaneous users.

state = {
    "url": None,       # The currently loaded website URL
    "title": None,     # Page title, used for display purposes
    "content": None,   # Extracted, trimmed website text
    "history": [],     # List of {"role": "user"/"assistant", "content": str}
}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LoadWebsiteRequest(BaseModel):
    url: str


class AskQuestionRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Website fetching & content extraction
# ---------------------------------------------------------------------------

def is_valid_url(url: str) -> bool:
    """Basic validation: the URL must have an http/https scheme and a host."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


def extract_text_from_html(html: str) -> tuple[str, str]:
    """
    Parse raw HTML and return (title, cleaned_text).

    Removes elements that are very unlikely to contain useful, readable
    content (scripts, styles, navigation, headers/footers, etc.) before
    pulling out the visible text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Grab the page title before stripping tags, for a nicer confirmation
    # message to the user.
    title = soup.title.get_text(strip=True) if soup.title else "Untitled page"

    # Strip out tags that generally hold no useful natural-language content.
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                      "aside", "form", "svg", "iframe"]):
        tag.decompose()

    # Extract remaining visible text, using newlines as separators so that
    # different blocks of text don't get glued together.
    raw_text = soup.get_text(separator="\n")

    # Collapse repeated blank lines/whitespace into a clean, compact string.
    lines = [line.strip() for line in raw_text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    cleaned_text = "\n".join(non_empty_lines)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    return title, cleaned_text


def fetch_website_content(url: str) -> tuple[str, str]:
    """
    Fetch the given URL and return (title, cleaned_text).
    Raises ValueError with a user-friendly message on any failure.
    """
    headers = {
        # Some sites block requests without a browser-like User-Agent.
        "User-Agent": (
            "Mozilla/5.0 (compatible; AIWebsiteChatAssistant/1.0; "
            "+https://example.local)"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        raise ValueError("The website took too long to respond. Please try again or use a different URL.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Could not connect to that website. Please check the URL and try again.")
    except requests.exceptions.RequestException:
        raise ValueError("Something went wrong while trying to reach that website.")

    if response.status_code >= 400:
        raise ValueError(
            f"The website returned an error (HTTP {response.status_code}). "
            "It may be unavailable or blocking automated requests."
        )

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text" not in content_type:
        raise ValueError("This URL does not appear to point to a readable web page (HTML).")

    title, text = extract_text_from_html(response.text)

    if not text or len(text.strip()) == 0:
        raise ValueError("No readable text content could be found on this page.")

    # Trim to keep the prompt within a safe size for the model's context window.
    trimmed_text = text[:MAX_CONTENT_CHARS]

    return title, trimmed_text


# ---------------------------------------------------------------------------
# Groq API call
# ---------------------------------------------------------------------------

# The system prompt is the guardrail: it forces the model to rely only on the
# supplied website content, forbids inventing information, and instructs it
# to say clearly when an answer isn't available in the content.
SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant that answers questions about ONE specific website.

Rules you must always follow:
- Base your answers only on the "Website content" provided below.
- Do not invent, assume, or add information that is not present in that content.
- If the answer is not present in the website content, say clearly that the information could not be found on the website. Do not guess.
- Keep answers concise and directly useful.
- Treat follow-up questions as continuing the conversation about this same website.

Website URL: {url}

Website content:
\"\"\"
{content}
\"\"\"
"""


def build_messages_for_groq(question: str) -> list[dict]:
    """Assemble the full message list (system + history + new question)."""
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(url=state["url"], content=state["content"]),
    }

    # Only keep the most recent turns to bound prompt size as chats grow.
    recent_history = state["history"][-(MAX_HISTORY_TURNS * 2):]

    messages = [system_message] + recent_history + [{"role": "user", "content": question}]
    return messages


def call_groq(messages: list[dict]) -> str:
    """
    Send a chat completion request to Groq and return the assistant's reply text.
    Raises ValueError with a user-friendly message on any failure.
    """
    if not GROQ_API_KEY:
        raise ValueError("The server is missing its GROQ_API_KEY configuration. Please contact the site administrator.")

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        # temperature=0 for deterministic, non-creative, fact-grounded answers.
        "temperature": 0,
        "max_tokens": MAX_ANSWER_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    except requests.exceptions.RequestException:
        raise ValueError("Could not reach the AI service. Please try again in a moment.")

    if response.status_code != 200:
        raise ValueError("The AI service returned an error while generating a response. Please try again.")

    try:
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        raise ValueError("Received an unexpected response from the AI service.")

    return answer.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page chat interface."""
    return HTML_PAGE


@app.post("/api/load-website")
def load_website(payload: LoadWebsiteRequest):
    """Validate, fetch, and extract content for the given URL; reset chat history."""
    url = payload.url.strip()

    if not url or not is_valid_url(url):
        return JSONResponse(
            status_code=400,
            content={"error": "Please enter a valid URL that starts with http:// or https://"},
        )

    try:
        title, content = fetch_website_content(url)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred while processing the website."},
        )

    # Store the new website context and reset the conversation history,
    # since a new website means a new topic of conversation.
    state["url"] = url
    state["title"] = title
    state["content"] = content
    state["history"] = []

    return {"message": f"Website loaded successfully: {title}", "title": title, "url": url}


@app.post("/api/ask")
def ask_question(payload: AskQuestionRequest):
    """Answer a question about the currently loaded website."""
    question = payload.question.strip()

    if not question:
        return JSONResponse(status_code=400, content={"error": "Please enter a question."})

    if not state["content"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Please load a website first before asking questions."},
        )

    messages = build_messages_for_groq(question)

    try:
        answer = call_groq(messages)
    except ValueError as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred while generating the answer."},
        )

    # Append this exchange to the conversation history for future follow-ups.
    state["history"].append({"role": "user", "content": question})
    state["history"].append({"role": "assistant", "content": answer})

    return {"answer": answer}


# ---------------------------------------------------------------------------
# Embedded frontend (single HTML page, no build step required)
# ---------------------------------------------------------------------------
# Design notes:
# - Palette: ink navy background, slate panels, hairline borders, warm amber
#   accent — chosen to feel like a focused "reading tool", not a generic
#   rounded-card SaaS template.
# - Monospace is used only for "chrome" (wordmark, URL bar, status pill) to
#   echo the idea of reading a raw URL/address bar; conversation text uses a
#   normal system sans-serif for readability.
# - Assistant replies often contain **bold**, tables, lists, and literal
#   "<br>" text from the model. formatMessage() below renders those into
#   real HTML instead of showing raw markdown symbols, without relying on
#   any external library (keeps the app self-contained).

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Website Chat Assistant</title>
<style>
  :root {
    --bg: #0F1418;
    --panel: #16202A;
    --panel-alt: #1B2733;
    --border: #26343F;
    --text: #E8EDF1;
    --text-dim: #8CA0AF;
    --accent: #E3B341;
    --accent-dim: #4A3D1E;
    --danger: #E06B6B;
    --mono: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    min-height: 100vh;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    display: flex;
    justify-content: center;
    padding: 32px 16px;
  }

  .app {
    width: 100%;
    max-width: 760px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* ---- Header ---- */
  .app-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
  }

  .wordmark {
    font-family: var(--mono);
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .wordmark .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
  }

  .site-pill {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 10px;
    max-width: 340px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: none;
  }

  .site-pill.visible { display: inline-block; }

  /* ---- URL / address bar ---- */
  .url-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px;
  }

  .url-bar .globe {
    width: 32px;
    height: 32px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 0.9rem;
  }

  #url-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.9rem;
    padding: 8px 4px;
    min-width: 0;
  }

  #url-input::placeholder { color: var(--text-dim); }

  button {
    font-family: var(--sans);
    font-size: 0.85rem;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
    cursor: pointer;
    transition: filter 0.15s ease, transform 0.05s ease;
  }

  button:active { transform: scale(0.98); }

  #load-btn, #ask-btn {
    background: var(--accent);
    color: #1A1404;
  }

  #load-btn:hover, #ask-btn:hover { filter: brightness(1.08); }

  #load-btn:disabled, #ask-btn:disabled {
    background: var(--border);
    color: var(--text-dim);
    cursor: not-allowed;
    filter: none;
  }

  /* ---- Status line ---- */
  #status {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--text-dim);
    min-height: 18px;
    padding-left: 4px;
  }

  #status.error { color: var(--danger); }
  #status.success { color: var(--accent); }

  /* ---- Chat panel ---- */
  #chat-section { display: none; flex-direction: column; gap: 12px; }

  #chat-log {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    height: 420px;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  #chat-log::-webkit-scrollbar { width: 8px; }
  #chat-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .msg { display: flex; }
  .msg.user { justify-content: flex-end; }
  .msg.assistant { justify-content: flex-start; }

  .bubble {
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 0.92rem;
    line-height: 1.55;
    word-wrap: break-word;
  }

  .msg.user .bubble {
    background: var(--accent-dim);
    border: 1px solid #5C4B24;
    color: #F5E6BE;
  }

  .msg.assistant .bubble {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--text);
  }

  .msg.assistant .bubble.error {
    border-color: var(--danger);
    color: var(--danger);
  }

  /* Rendered markdown inside assistant bubbles */
  .bubble p { margin: 0 0 8px 0; }
  .bubble p:last-child { margin-bottom: 0; }
  .bubble strong { color: var(--accent); font-weight: 700; }
  .bubble ul { margin: 4px 0 8px 0; padding-left: 20px; }
  .bubble li { margin-bottom: 4px; }

  .bubble table {
    border-collapse: collapse;
    width: 100%;
    margin: 6px 0 10px 0;
    font-size: 0.85rem;
  }
  .bubble th, .bubble td {
    border: 1px solid var(--border);
    padding: 6px 8px;
    text-align: left;
  }
  .bubble th {
    background: #22303B;
    color: var(--accent);
    font-weight: 600;
  }

  /* ---- Question input ---- */
  #question-section {
    display: flex;
    gap: 8px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px;
  }

  #question-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: var(--sans);
    font-size: 0.92rem;
    padding: 8px 10px;
    min-width: 0;
  }

  #question-input::placeholder { color: var(--text-dim); }

  /* Typing indicator while waiting for the AI */
  .typing-dots {
    display: inline-flex;
    gap: 4px;
    align-items: center;
    padding: 2px 0;
  }
  .typing-dots span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-dim);
    animation: blink 1.2s infinite ease-in-out;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0.25; } 40% { opacity: 1; } }

  @media (max-width: 480px) {
    #chat-log { height: 60vh; }
    .bubble { max-width: 92%; }
  }
</style>
</head>
<body>

<div class="app">

  <div class="app-header">
    <div class="wordmark"><span class="dot"></span>AI Website Chat Assistant</div>
    <div class="site-pill" id="site-pill"></div>
  </div>

  <div class="url-bar">
    <div class="globe">://</div>
    <input id="url-input" type="text" placeholder="https://example.com" />
    <button id="load-btn">Load Website</button>
  </div>

  <div id="status"></div>

  <div id="chat-section">
    <div id="chat-log"></div>
    <div id="question-section">
      <input id="question-input" type="text" placeholder="Ask a question about this website..." />
      <button id="ask-btn">Send</button>
    </div>
  </div>

</div>

<script>
const urlInput = document.getElementById('url-input');
const loadBtn = document.getElementById('load-btn');
const statusDiv = document.getElementById('status');
const sitePill = document.getElementById('site-pill');
const chatSection = document.getElementById('chat-section');
const chatLog = document.getElementById('chat-log');
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');

// Escape raw HTML so user/model text can never inject markup or scripts.
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Turn a block of "|"-delimited lines into an HTML table, if it looks like one.
// Returns the table HTML, or null if the block isn't actually a table.
function renderTableBlock(lines) {
  const rows = lines
    .map(l => l.trim())
    .filter(l => l.length > 0)
    .map(l => l.replace(/^\\|/, '').replace(/\\|$/, '').split('|').map(c => c.trim()));

  if (rows.length < 1) return null;

  // A separator row like "---|---|---" (optionally with colons) is optional;
  // skip it if present, otherwise treat the first row as the header.
  const isSeparator = (row) => row.every(cell => /^:?-+:?$/.test(cell));
  const header = rows[0];
  const bodyRows = (rows.length > 1 && isSeparator(rows[1])) ? rows.slice(2) : rows.slice(1);

  let html = '<table><thead><tr>';
  header.forEach(cell => { html += `<th>${inlineFormat(cell)}</th>`; });
  html += '</tr></thead><tbody>';
  bodyRows.forEach(row => {
    html += '<tr>';
    row.forEach(cell => { html += `<td>${inlineFormat(cell)}</td>`; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  return html;
}

// Apply inline formatting (bold, literal <br> markers) to a single line of
// already-escaped text.
function inlineFormat(escapedText) {
  return escapedText
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')   // **bold** -> <strong>
    .replace(/&lt;br\\s*\\/?&gt;/gi, '<br>');                // literal <br> -> real line break
}

// Convert the model's lightweight markdown (bold, tables, bullet lists,
// paragraphs) into safe HTML for display. Text is escaped first, so this
// never introduces real script/markup from the model or the user.
function formatMessage(rawText) {
  const escaped = escapeHtml(rawText);
  const lines = escaped.split('\\n');

  let html = '';
  let paragraphBuffer = [];
  let listBuffer = [];
  let tableBuffer = [];

  const flushParagraph = () => {
    if (paragraphBuffer.length) {
      html += `<p>${paragraphBuffer.join('<br>')}</p>`;
      paragraphBuffer = [];
    }
  };
  const flushList = () => {
    if (listBuffer.length) {
      html += '<ul>' + listBuffer.map(item => `<li>${inlineFormat(item)}</li>`).join('') + '</ul>';
      listBuffer = [];
    }
  };
  const flushTable = () => {
    if (tableBuffer.length) {
      const table = renderTableBlock(tableBuffer);
      if (table) html += table;
      tableBuffer = [];
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.includes('|') && trimmed.split('|').length > 2) {
      // Part of a table block.
      flushParagraph();
      flushList();
      tableBuffer.push(trimmed);
      continue;
    }
    flushTable();

    if (/^[-*]\\s+/.test(trimmed)) {
      // Bullet list item.
      flushParagraph();
      listBuffer.push(trimmed.replace(/^[-*]\\s+/, ''));
      continue;
    }
    flushList();

    if (trimmed === '') {
      flushParagraph();
      continue;
    }

    paragraphBuffer.push(inlineFormat(trimmed));
  }
  flushParagraph();
  flushList();
  flushTable();

  return html || `<p>${escaped}</p>`;
}

function addMessage(role, text, isError = false) {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble' + (isError ? ' error' : '');
  bubble.innerHTML = role === 'assistant' ? formatMessage(text) : escapeHtml(text);
  wrapper.appendChild(bubble);
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrapper;
}

function addTypingIndicator() {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg assistant';
  wrapper.id = 'typing-indicator';
  wrapper.innerHTML = '<div class="bubble"><span class="typing-dots"><span></span><span></span><span></span></span></div>';
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function setBusy(busy) {
  loadBtn.disabled = busy;
  askBtn.disabled = busy;
}

loadBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  if (!url) return;

  setBusy(true);
  statusDiv.textContent = 'Loading website...';
  statusDiv.className = '';
  sitePill.className = 'site-pill';

  try {
    const res = await fetch('/api/load-website', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();

    if (!res.ok) {
      statusDiv.textContent = data.error || 'Failed to load website.';
      statusDiv.className = 'error';
      chatSection.style.display = 'none';
      return;
    }

    statusDiv.textContent = 'Ready — ask a question below.';
    statusDiv.className = 'success';
    sitePill.textContent = data.title;
    sitePill.className = 'site-pill visible';
    chatLog.innerHTML = '';
    chatSection.style.display = 'flex';
    questionInput.focus();
  } catch (err) {
    statusDiv.textContent = 'Could not reach the server.';
    statusDiv.className = 'error';
  } finally {
    setBusy(false);
  }
});

askBtn.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage('user', question);
  questionInput.value = '';
  setBusy(true);
  addTypingIndicator();

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    removeTypingIndicator();

    if (!res.ok) {
      addMessage('assistant', data.error || 'Something went wrong.', true);
      return;
    }
    addMessage('assistant', data.answer);
  } catch (err) {
    removeTypingIndicator();
    addMessage('assistant', 'Could not reach the server.', true);
  } finally {
    setBusy(false);
  }
});

questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') askBtn.click();
});
urlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') loadBtn.click();
});
</script>

</body>
</html>
"""