const state = {
  busy: false,
};

const els = {
  messages: document.querySelector("#messages"),
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendButton"),
  toolList: document.querySelector("#toolList"),
  grepList: document.querySelector("#grepList"),
  vecList: document.querySelector("#vecList"),
};

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.input.disabled = busy;
}

function appendMessage(role, label, text) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = `<span class="message-label"></span><div></div>`;
  node.querySelector(".message-label").textContent = label;
  node.querySelector("div").textContent = text;
  els.messages.append(node);
  els.messages.scrollTop = els.messages.scrollHeight;
}

// ---------------------------------------------------------------------------
// Right panel renderers
// ---------------------------------------------------------------------------

function renderTool(toolUsed, toolReasoning, toolCommands, grepLatency, vecLatency) {
  els.toolList.innerHTML = "";
  els.toolList.classList.remove("empty");

  if (toolUsed === "none" || !toolUsed) {
    els.toolList.classList.add("empty");
    const li = document.createElement("li");
    li.textContent = "No tool was needed for this turn.";
    els.toolList.append(li);
    return;
  }

  const fetchCmd = (toolCommands || []).find(c => c && c.startsWith("ARGET"));
  const hasGrep = (toolCommands || []).some(c => c && c.startsWith("ARGREP"));
  const fetchLabel = fetchCmd && fetchCmd.startsWith("ARGETRANGE") ? "Array Get Range" : "Array Get";
  let bothLabel = "Grep + Vector";
  if (toolUsed === "both") {
    bothLabel = hasGrep ? "Grep + Vector" : `${fetchLabel} + Vector`;
  }
  const grepFetchLabel = `Array Grep + ${fetchLabel}`;
  const toolLabel = { grep: "Array Grep", vector: "Vector Search", fetch: fetchLabel, both: bothLabel, grep_fetch: grepFetchLabel, arlen: "Array Len" }[toolUsed] || toolUsed;

  // Plain text rows
  const rows = [`Tool: ${toolLabel}`];
  if (toolReasoning) rows.push(`Reason: ${toolReasoning}`);

  for (const text of rows) {
    const li = document.createElement("li");
    li.textContent = text;
    els.toolList.append(li);
  }

  // Command rows — one per tool called (filter blanks, use textContent for safety)
  for (const cmd of (toolCommands || []).filter(c => c && c.trim())) {
    const li = document.createElement("li");
    li.className = "command-row";
    const label = document.createTextNode("Command: ");
    const code = document.createElement("code");
    code.textContent = cmd;
    li.append(label, code);
    els.toolList.append(li);
  }
}

function truncate(text, max = 72) {
  return text.length > max ? text.slice(0, max) + "…" : text;
}

function formatLatency(ms) {
  if (ms < 1) {
    return `${Math.round(ms * 1000)}µs`;
  }
  if (ms < 10) {
    return `${ms.toFixed(1)}ms`;
  }
  return `${Math.round(ms)}ms`;
}

function setResultMeta(sectionEl, latencyMs) {
  sectionEl.querySelectorAll(".result-meta").forEach(el => el.remove());
  if (latencyMs == null) return;

  const meta = document.createElement("div");
  meta.className = "result-meta";
  const lat = document.createElement("span");
  lat.className = "result-latency";
  lat.textContent = formatLatency(latencyMs);
  meta.append(lat);
  sectionEl.querySelector(".section-heading").append(meta);
}

function renderGrepResults(results, latency, toolUsed, toolCommands) {
  els.grepList.innerHTML = "";
  els.grepList.classList.remove("empty", "grep-active");
  // Retitle section to reflect the actual Redis command used
  const sectionEl = els.grepList.closest(".retrieval-section");
  const titleEl = sectionEl.querySelector("h2");
  titleEl.textContent = "Array Command Result";
  setResultMeta(sectionEl, latency);

  if (!results || results.length === 0) {
    els.grepList.classList.add("empty");
    const li = document.createElement("li");
    li.textContent = "Not used this turn.";
    els.grepList.append(li);
    return;
  }

  els.grepList.classList.add("grep-active");

  for (const r of results) {
    const li = document.createElement("li");
    const lineNum = document.createElement("span");
    lineNum.className = "line-num";
    lineNum.textContent = `L${r.line}`;
    const text = document.createElement("span");
    text.className = "row-text";
    text.textContent = truncate(r.content);
    li.append(lineNum, text);
    els.grepList.append(li);
  }
}

function renderVectorResults(results, latency) {
  els.vecList.innerHTML = "";
  els.vecList.classList.remove("empty", "vec-active");
  setResultMeta(els.vecList.closest(".retrieval-section"), latency);

  if (!results || results.length === 0) {
    els.vecList.classList.add("empty");
    const li = document.createElement("li");
    li.textContent = "Not used this turn.";
    els.vecList.append(li);
    return;
  }

  els.vecList.classList.add("vec-active");
  // Filter code-fence markers and other markdown noise that got indexed
  const displayResults = results.filter(r => {
    const t = (r.content || "").trim();
    return t.length > 0 && !/^`+$/.test(t);
  });
  for (const r of displayResults) {
    const li = document.createElement("li");
    li.append(document.createTextNode(truncate(r.content)));
    els.vecList.append(li);
  }
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return payload;
}

async function sendMessage(message) {
  setBusy(true);
  appendMessage("user", "👤 You", message);
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    appendMessage("ai", "🤖 Agent", payload.assistant_message);
    renderTool(payload.tool_used, payload.tool_reasoning, payload.tool_commands, payload.total_latency_ms, payload.vector_latency_ms);
    renderGrepResults(payload.grep_results, payload.total_latency_ms, payload.tool_used, payload.tool_commands);
    renderVectorResults(payload.vector_results, payload.vector_latency_ms);
  } catch (error) {
    appendMessage("system", "Error", error.message);
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = els.input.value.trim();
  if (!message || state.busy) return;
  els.input.value = "";
  sendMessage(message);
});
