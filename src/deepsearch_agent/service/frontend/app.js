"use strict";
/* ---------- 基础设施 ---------- */
const tokenKey = "deepsearch.token";
const $ = (id) => document.getElementById(id);
let lastSeq = 0;        // SSE 去重水位：重连/回放都只会重发旧 seq，单调消费即可
let streamAbort = null; // 当前 SSE 的 AbortController

function authHeaders(extra) {
  const t = localStorage.getItem(tokenKey);
  return Object.assign(t ? { Authorization: "Bearer " + t } : {}, extra || {});
}

async function api(path, options) {
  const resp = await fetch(path, Object.assign({ headers: authHeaders({ "Content-Type": "application/json" }) }, options));
  if (resp.status === 401 && path !== "/api/login" && path !== "/api/register") {
    logout();
    throw new Error("登录已过期，请重新登录。");
  }
  const body = resp.status === 204 ? null : await resp.json().catch(() => null);
  if (!resp.ok) {
    throw new Error((body && body.detail) || ("请求失败（" + resp.status + "）"));
  }
  return body;
}

function logout() {
  localStorage.removeItem(tokenKey);
  if (streamAbort) streamAbort.abort();
  location.hash = "#/";
  showView();
}

function show(el) { document.querySelectorAll("main > section").forEach(s => s.classList.add("hide")); el.classList.remove("hide"); }

/* ---------- 视图路由（hash 极简版） ---------- */
async function showView() {
  // 非路由 hash（锚点或外链分享）：归一化回 #/ 再正常路由，避免白屏或误切视图。
  if (location.hash && !location.hash.startsWith("#/")) {
    history.replaceState(null, "", location.pathname + location.search + "#/");
  }
  const m = location.hash.match(/^#\/run\/(.+)$/);
  const loggedIn = !!localStorage.getItem(tokenKey);
  if (m) { renderRun(m[1]); return; }
  if (!loggedIn) { show($("view-auth")); return; }
  show($("view-home"));
  try {
    const me = await api("/api/me");
    $("whoami").textContent = me.email;
    $("logout").classList.remove("hide");
    await refreshRuns();
  } catch (e) { $("whoami").textContent = ""; }
}
window.addEventListener("hashchange", showView);

/* ---------- 登录/注册 ---------- */
let authMode = "login";
function setAuthMode(mode) {
  authMode = mode;
  $("auth-submit").textContent = mode === "login" ? "登录" : "注册";
  $("tab-login").disabled = mode === "login";
  $("tab-register").disabled = mode !== "login";
  $("auth-error").textContent = "";
}
$("tab-login").onclick = () => setAuthMode("login");
$("tab-register").onclick = () => setAuthMode("register");
$("auth-submit").onclick = async () => {
  $("auth-error").textContent = "";
  try {
    const r = await api("/api/" + authMode, {
      method: "POST",
      body: JSON.stringify({ email: $("auth-email").value, password: $("auth-password").value }),
    });
    localStorage.setItem(tokenKey, r.token);
    location.hash = "#/";
    showView();
  } catch (e) { $("auth-error").textContent = e.message; }
};
$("auth-password").addEventListener("keydown", (e) => { if (e.key === "Enter") $("auth-submit").click(); });
$("logout").onclick = logout;

/* ---------- 首页 ---------- */
$("submit").onclick = async () => {
  const query = $("query").value.trim();
  $("home-error").textContent = "";
  if (!query) { $("home-error").textContent = "问题不能为空。"; return; }
  $("submit").disabled = true;
  try {
    const r = await api("/api/runs", { method: "POST", body: JSON.stringify({ query }) });
    location.hash = "#/run/" + r.run_id;
  } catch (e) { $("home-error").textContent = e.message; }
  $("submit").disabled = false;
};
$("query").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") $("submit").click();
});

function fmtAgo(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 60) return "刚刚";
  if (s < 3600) return Math.floor(s / 60) + " 分钟前";
  if (s < 86400) return Math.floor(s / 3600) + " 小时前";
  if (s < 86400 * 7) return Math.floor(s / 86400) + " 天前";
  return (d.getMonth() + 1) + "月" + d.getDate() + "日";
}

async function refreshRuns() {
  let runs = [];
  try { runs = await api("/api/runs"); } catch (e) { return; }
  cachedRuns = runs;
  updateRunsPerPage();
  const pageCount = Math.ceil(runs.length / runsPerPage);
  runsPage = Math.min(runsPage, Math.max(0, pageCount - 1));
  renderRunsPage();
}

let runsPerPage = 10;
let cachedRuns = [];
let runsPage = 0;

function preferredRunsPerPage() {
  const listTop = $("runs-list").getBoundingClientRect().top;
  const available = window.innerHeight - listTop;
  if (available >= 900) return 20;
  if (available >= 650) return 15;
  return 10;
}

function updateRunsPerPage() {
  const next = preferredRunsPerPage();
  if (next === runsPerPage) return;
  const firstVisibleRun = runsPage * runsPerPage;
  runsPerPage = next;
  runsPage = Math.floor(firstVisibleRun / runsPerPage);
}

function renderRunsPage() {
  const runs = cachedRuns;
  const list = $("runs-list");
  list.innerHTML = "";
  $("runs-empty").classList.toggle("hide", runs.length > 0);
  $("runs-count").textContent = runs.length ? "共 " + runs.length + " 条" : "";
  const pageCount = Math.ceil(runs.length / runsPerPage);
  const start = runsPage * runsPerPage;
  for (const run of runs.slice(start, start + runsPerPage)) {
    const row = document.createElement("div");
    row.className = "run-row";
    const badge = badgeFor(run.status, run);
    const b = document.createElement("span");
    b.className = "badge " + badge.cls; b.textContent = badge.text;
    const q = document.createElement("span");
    q.className = "run-q"; q.textContent = run.query; q.title = run.query;
    const counts = document.createElement("span");
    counts.className = "run-counts";
    counts.textContent = (run.evidence_count || run.source_count)
      ? "证据 " + run.evidence_count + " · 来源 " + run.source_count : "";
    const time = document.createElement("span");
    time.className = "run-time"; time.textContent = fmtAgo(run.created_at);
    row.append(b, q, counts, time);
    row.onclick = () => { location.hash = "#/run/" + run.id; };
    list.appendChild(row);
  }

  const pager = $("runs-pager");
  pager.classList.toggle("hide", pageCount <= 1);
  $("runs-prev").disabled = runsPage === 0;
  $("runs-next").disabled = runsPage >= pageCount - 1;
  const dots = $("runs-page-dots");
  dots.innerHTML = "";
  for (let page = 0; page < pageCount; page++) {
    const dot = document.createElement("button");
    dot.className = "runs-page-dot" + (page === runsPage ? " active" : "");
    dot.setAttribute("aria-label", "第 " + (page + 1) + " 页");
    dot.title = "第 " + (page + 1) + " 页";
    dot.onclick = () => { runsPage = page; renderRunsPage(); };
    dots.appendChild(dot);
  }
}
$('runs-prev').onclick = () => { if (runsPage > 0) { runsPage--; renderRunsPage(); } };
$('runs-next').onclick = () => {
  if ((runsPage + 1) * runsPerPage < cachedRuns.length) { runsPage++; renderRunsPage(); }
};
let runsResizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(runsResizeTimer);
  runsResizeTimer = setTimeout(() => {
    const previous = runsPerPage;
    updateRunsPerPage();
    if (previous !== runsPerPage && !$("view-home").classList.contains("hide")) renderRunsPage();
  }, 150);
});
setInterval(() => { if ($("view-home").classList.contains("hide") === false) refreshRuns(); }, 5000);

/* ---------- 进度呈现：全局时间线（supervisor 整体视角）+ 方向卡片（searcher 细节） ---------- */
function progressLine(text, warn) {
  const div = document.createElement("div");
  div.className = "ln" + (warn ? " warn" : "");
  div.textContent = text;
  const log = $("global-log");
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

const taskCards = new Map();
const stageBlocks = new Map();
let taskSeq = 0;

/* 阶段块按需创建：事件到达才出现（解决"还没执行框先出现"）；
   running 绿点 → done 灰点 / failed 红点，三态不再复用灰色。 */
function ensureStage(stage, title) {
  let block = stageBlocks.get(stage);
  if (!block) {
    block = document.createElement("div");
    block.className = "stage-block running";
    const head = document.createElement("div"); head.className = "stage-head";
    const dot = document.createElement("span"); dot.className = "dot";
    const t = document.createElement("span"); t.className = "stage-title";
    const count = document.createElement("span"); count.className = "stage-count";
    head.append(dot, t, count);
    const body = document.createElement("div"); body.className = "stage-body";
    block.append(head, body);
    $("stages").appendChild(block);
    stageBlocks.set(stage, block);
  }
  if (title) block.querySelector(".stage-title").textContent = title;
  else if (!block.querySelector(".stage-title").textContent) {
    block.querySelector(".stage-title").textContent = stage;
  }
  return block;
}

/* token 预览：delta 追加进 tentative 行（斜体）；该 stage 的聚合帧（带 seq、
   已落库、可回放）一旦到达即删除预览行、换成正式文本。掉线丢 delta 无碍。 */
const tentative = new Map();

function deltaAppend(channel, text) {
  if (!text) return;
  const block = ensureStage(channel);
  let t = tentative.get(channel);
  if (!t) {
    const el = document.createElement("div");
    el.className = "ln tentative";
    block.querySelector(".stage-body").appendChild(el);
    t = { el, text: "" };
    tentative.set(channel, t);
  }
  t.text += text;
  t.el.textContent = t.text;
  const body = block.querySelector(".stage-body");
  if (body.lastChild !== t.el) body.appendChild(t.el);  // 聚合行插入后预览保持在底
}

function finalizeTentative(stage) {
  const t = tentative.get(stage);
  if (t) { t.el.remove(); tentative.delete(stage); }
}

function clearAllTentative() {
  for (const stage of [...tentative.keys()]) finalizeTentative(stage);
}

function stageNote(stage, text, warn) {
  const t = tentative.get(stage);
  if (t) {
    if (text && t.text.length > text.length) {
      // 聚合帧比流式预览短（事件侧有界化必然存在）：晋升预览为正式行，
      // 不做"打字完成后突然截短"的替换。
      t.el.classList.remove("tentative");
      if (warn) t.el.classList.add("warn");
      tentative.delete(stage);
      return;
    }
    t.el.remove();
    tentative.delete(stage);
  }
  if (!text) return;
  const body = ensureStage(stage).querySelector(".stage-body");
  const div = document.createElement("div");
  div.className = "ln" + (warn ? " warn" : "");
  div.textContent = text;
  body.appendChild(div);
  const lines = body.querySelectorAll(":scope > .ln");
  if (lines.length > 40) lines[0].remove();
}

function ensureCard(id, title) {
  let card = taskCards.get(id);
  if (card) {
    if (title) {
      const el = card.querySelector(".tcard-title");
      el.textContent = card.dataset.label + title;
      el.title = title;
    }
    return card;
  }
  const label = "方向 " + (++taskSeq) + " · ";
  card = document.createElement("div");
  card.className = "tcard running";
  card.dataset.label = label;
  const head = document.createElement("div"); head.className = "tcard-head";
  const t = document.createElement("div"); t.className = "tcard-title";
  t.textContent = label + (title || "研究方向");
  t.title = title || "";
  const toggle = document.createElement("button"); toggle.className = "tcard-toggle"; toggle.textContent = "▸";
  toggle.title = "展开动作明细";
  toggle.onclick = () => card.classList.toggle("expanded");
  head.append(t, toggle);
  const live = document.createElement("div"); live.className = "tcard-live"; live.textContent = "准备中…";
  const details = document.createElement("ul"); details.className = "tcard-details";
  card.append(head, live, details);
  ensureStage("supervisor").querySelector(".stage-body").appendChild(card);
  taskCards.set(id, card);
  updateTaskCount();
  return card;
}

function updateTaskCount() {
  const supervisor = stageBlocks.get("supervisor");
  if (!supervisor) return;
  const running = supervisor.querySelector(".tcard.running");
  supervisor.querySelector(".stage-count").textContent =
    taskSeq ? "已派 " + taskSeq + " 个方向" + (running ? " · 执行中" : "") : "";
}

function renderStats(data) {
  const el = $("run-metrics");
  el.innerHTML = "";
  const chips = [
    "第 " + data.round + " 轮",
    "方向 " + data.tasks_completed + "/" + data.tasks_total,
    "证据 +" + data.evidence_added + "（累计 " + data.evidence_total + "）",
  ];
  for (const text of chips) {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = text;
    el.appendChild(span);
  }
}

function cardAction(card, text) {
  const live = card.querySelector(".tcard-live");
  live.textContent = text;
  live.classList.remove("anim"); void live.offsetWidth; live.classList.add("anim");
  const details = card.querySelector(".tcard-details");
  const last = details.lastElementChild;
  if (!last || last.textContent !== text) {          // 连续重复动作只留一条
    const li = document.createElement("li"); li.textContent = text;
    details.appendChild(li);
    while (details.children.length > 80) details.removeChild(details.firstChild);
  }
  details.scrollTop = details.scrollHeight;
}

function handleFrame(event, data) {
  if (typeof data.seq === "number") {
    if (data.seq <= lastSeq) return; // 回放/重连去重
    lastSeq = data.seq;
  }
  if (event === "text_delta") deltaAppend(data.channel, data.text);
  else if (event === "tick") progressLine(data.text);
  else if (event === "stage_open") ensureStage(data.stage, data.title);
  else if (event === "plan") stageNote(data.stage, data.text);
  else if (event === "stage_done") {
    const block = ensureStage(data.stage);
    block.classList.remove("running");
    block.classList.add(data.status === "failed" ? "failed" : "done");
    stageNote(data.stage, data.text);
  } else if (event === "error") progressLine(data.text, true);
  else if (event === "status") setStatus(data.status);
  else if (event === "stats") renderStats(data);
  else if (event === "task_open") ensureCard(data.task, data.title);
  else if (event === "task_update") {
    const card = ensureCard(data.task, "");           // 孤儿 update 兜底建卡
    if (data.text) cardAction(card, data.text);
  } else if (event === "task_done") {
    const card = ensureCard(data.task, "");
    card.classList.remove("running");
    card.classList.add(data.status === "failed" ? "failed" : data.status === "warn" ? "warn" : "done");
    cardAction(card, (data.status === "failed" ? "✗ " : "✓ ") + (data.summary || "本步完成"));
    updateTaskCount();
  } else if (event === "clarification") {
    renderClarification(data);
  } else if (event === "done") { clearAllTentative(); setStatus(data.status, data); onRunDone(data); }
}

function renderClarification(data) {
  if (streamAbort) { streamAbort.abort(); streamAbort = null; }
  $("clarification-card").classList.remove("hide");
  $("clarification-question").textContent = data.question || "请补充研究范围。";
  const box = $("clarification-options"); box.innerHTML = "";
  for (const [i, option] of (data.options || []).slice(0, 3).entries()) {
    const label = document.createElement("label");
    const radio = document.createElement("input");
    radio.type = "radio"; radio.name = "clarification-choice"; radio.value = option;
    if (i === 0) radio.checked = true;
    const text = document.createElement("span"); text.textContent = option;
    label.append(radio, text); box.appendChild(label);
  }
  $("clarification-other").value = "";
  $("clarification-submit").disabled = false;
  setStatus("awaiting_input");
}

function resetProgressView() {
  taskCards.clear();
  stageBlocks.clear();
  tentative.clear();
  taskSeq = 0;
  $("stages").innerHTML = "";
  $("global-log").innerHTML = "";
  $("run-metrics").innerHTML = "";
  $("clarification-card").classList.add("hide");
  $("clarification-other").value = "";
  $("clarification-error").textContent = "";
}

/* completed 只代表"产出了终稿"；writer 耗尽/研究不完美的残局报告降级显示，
   不再用一个绿色 completed 掩盖"没成功"。数据源：answer_mode/terminal_reason。 */
const STATUS_LABEL = {
  running: "进行中", resuming: "恢复续跑中", queued: "排队中", interrupted: "已中断，等待恢复", awaiting_input: "等待回答",
  completed: "已完成", failed: "失败", cancelled: "已取消",
};

function badgeFor(status, info) {
  if (status === "completed" && info && info.answer_mode === "review_limited") {
    return { cls: "degraded", text: "受限报告" };
  }
  const degraded = status === "completed" && info &&
    (info.answer_mode === "research_incomplete" || info.terminal_reason === "writer_exhausted");
  if (degraded) return { cls: "degraded", text: "部分报告" };
  return { cls: status, text: STATUS_LABEL[status] || status };
}

function setStatus(status, info) {
  const badge = $("run-status");
  const view = badgeFor(status, info);
  badge.className = "badge " + view.cls;
  badge.textContent = view.text;
  $("run-cancel").disabled = ["completed", "failed", "cancelled"].includes(status);
}

async function onRunDone(data) {
  if (streamAbort) { streamAbort.abort(); streamAbort = null; }
  $("report-card").classList.remove("hide");
  let detail;
  try { detail = await api("/api/runs/" + currentRunId); } catch (e) { return; }
  // 用完整 detail 重设徽章：done 帧缺 terminal_reason，只有这里能判"部分报告"。
  setStatus(detail.status, detail);
  if (detail.status === "awaiting_input" && detail.clarification) {
    renderClarification(detail.clarification);
    return;
  }
  const md = detail.report_markdown || detail.error_message || "（无报告内容）";
  renderReport(md, detail.citations || [], detail);
}

/* 渲染器拼在 markdown 里的报告头（# 研究报告 / ## 研究问题 / > 横幅）与页面
   头部卡片信息重复——剥出正文，统计信息改为独立的元提示栏（数字取 detail 字段）。 */
function stripReportHead(md) {
  const lines = md.split("\n");
  if (!/^#\s*研究报告/.test(lines[0] || "")) return md;
  let i = 1;
  while (i < lines.length && !lines[i].trim()) i++;
  if (/^##\s*研究问题/.test(lines[i] || "")) {
    i++;
    while (i < lines.length && lines[i].trim()) i++;
    while (i < lines.length && !lines[i].trim()) i++;
  }
  while (i < lines.length && lines[i].startsWith(">")) i++;   // 管道横幅
  return lines.slice(i).join("\n").replace(/^\n+/, "");
}

function renderReportMeta(detail) {
  const el = $("report-meta");
  el.innerHTML = "";
  if (!detail || !detail.report_markdown) { el.classList.add("hide"); return; }
  el.classList.remove("hide", "warn");
  const parts = [];
  const degraded = detail.answer_mode === "research_incomplete"
    || detail.terminal_reason === "writer_exhausted";
  if (detail.answer_mode === "quick_answer") {
    parts.push("即时回答 · 未联网检索");
  } else {
    if (detail.evidence_count) parts.push("Evidence " + detail.evidence_count);
    if (detail.source_count) parts.push("来源 " + detail.source_count);
  }
  if (detail.answer_mode === "review_limited") {
    el.classList.add("warn");
    parts.unshift("受限报告 · 已达到审阅修订上限");
  } else if (degraded) {
    el.classList.add("warn");
    parts.unshift("部分报告");
  }
  for (const text of parts) {
    const span = document.createElement("span");
    span.textContent = text;
    el.appendChild(span);
  }
}

/* 参考来源表由管道生成、编号权威唯一：正文 [来源N] 与表条目一一对应。
   前端把报告拆成正文/参考表两段——正文链接化渲染，来源面板带序号与锚点 id，
   点击 [来源N] 即跳转。citations_json 仅作解析失败时的无编号兜底。 */
function splitReport(md) {
  const m = md.match(/^## 参考来源[ \t]*$/m);
  if (!m) return [md, null];
  return [md.slice(0, m.index), md.slice(m.index + m[0].length)];
}

function parseReferences(refMd) {
  const items = [];
  for (const raw of refMd.split("\n")) {
    const line = raw.trim();
    let m = line.match(/^- \[(来源(\d+))\]\s*(.*?):\s*(\S+)\s*$/);
    if (m) {
      items.push({ label: m[1], num: Number(m[2]), title: m[3], url: m[4], quote: "" });
      continue;
    }
    m = line.match(/^- \[(来源(\d+))\]\s*(\S+)\s*$/);  // 无标题变体
    if (m) { items.push({ label: m[1], num: Number(m[2]), title: m[4], url: m[4], quote: "" }); continue; }
    m = line.match(/^>\s*「([\s\S]*)」\s*$/);
    if (m && items.length) items[items.length - 1].quote = m[1];
  }
  return items;
}

function renderReport(md, fallbackCitations, detail) {
  const stripped = stripReportHead(md);
  const [bodyMd, refMd] = splitReport(stripped);
  renderReportMeta(detail);
  $("report-actions").classList.toggle("hide", !(detail && detail.report_markdown));
  const reportEl = $("report");
  if (window.marked && !window.__marked_failed) {
    // 论文式角标：[来源N] → <sup>[N]</sup> 可点击上标（marked 透传内联 HTML；
    // 渲染器只在散文里替换标记，代码区不含 [来源N]，无注入面）。
    const linkified = bodyMd.replace(
      /\[来源(\d+)\]/g,
      (_s, n) => `<sup class="cite-ref"><a href="#src-${n}">[${n}]</a></sup>`,
    );
    reportEl.innerHTML = marked.parse(linkified);
  } else {
    const pre = document.createElement("pre"); pre.textContent = stripped;
    reportEl.replaceChildren(pre);
  }

  const sourcesEl = $("sources");
  sourcesEl.innerHTML = "";
  const items = refMd ? parseReferences(refMd) : [];
  if (!items.length) {
    $("sources-title").classList.toggle("hide", !fallbackCitations.length);
    for (const c of fallbackCitations) {          // 兜底：老数据无表格式时列原始引用
      const p = document.createElement("p");
      p.textContent = c.title + " — ";
      const a = document.createElement("a");
      a.href = c.url; a.target = "_blank"; a.rel = "noopener"; a.textContent = c.url;
      p.appendChild(a);
      if (c.quote) { const q = document.createElement("i"); q.textContent = "「" + c.quote + "」"; p.appendChild(document.createElement("br")); p.appendChild(q); }
      sourcesEl.appendChild(p);
    }
    return;
  }
  $("sources-title").classList.remove("hide");
  for (const item of items) {
    const div = document.createElement("div");
    div.className = "src-item";
    div.id = "src-" + item.num;                   // 正文 [来源N] 锚点落点
    const head = document.createElement("div");
    const label = document.createElement("b"); label.textContent = "[" + item.label + "] ";
    head.appendChild(label);
    if (item.title && item.title !== item.url) head.appendChild(document.createTextNode(item.title + " — "));
    const a = document.createElement("a");
    a.href = item.url; a.target = "_blank"; a.rel = "noopener"; a.textContent = item.url;
    head.appendChild(a);
    div.appendChild(head);
    if (item.quote) {
      const q = document.createElement("blockquote");
      q.textContent = "「" + item.quote + "」";
      div.appendChild(q);
    }
    sourcesEl.appendChild(div);
  }
}

function reportExportText() {
  const parts = ["研究报告"];
  const body = $("report").innerText.trim();
  const sources = $("sources").innerText.trim();
  if (body) parts.push(body);
  if (sources && !$("sources-title").classList.contains("hide")) {
    parts.push("参考来源\n\n" + sources);
  }
  return parts.join("\n\n");
}

async function writeClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  const copied = document.execCommand("copy");
  area.remove();
  if (!copied) throw new Error("浏览器拒绝了剪贴板访问。");
}

function reportPdfFilename() {
  const query = ($("run-query").textContent || "研究报告")
    .trim()
    .replace(/[\\/:*?"<>|]/g, "-")
    .replace(/\s+/g, " ")
    .slice(0, 48);
  return (query || "研究报告") + ".pdf";
}

async function showActionResult(button, pendingText, successText, action) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = pendingText;
  try {
    await action();
    button.textContent = successText;
  } catch (error) {
    progressLine(error && error.message ? error.message : "操作失败，请稍后重试。", true);
    button.textContent = "重试";
  } finally {
    window.setTimeout(() => {
      button.textContent = original;
      button.disabled = false;
    }, 1500);
  }
}

$("report-copy").onclick = () => showActionResult(
  $("report-copy"),
  "复制中…",
  "已复制",
  () => writeClipboard(reportExportText()),
);

function createPdfExportNode() {
  // 不直接截图页面中的卡片：它带有当前屏幕位置、宽度和交互按钮，容易产生
  // 首页空白及错误分页。固定尺寸副本只承载报告和参考来源。
  const host = document.createElement("div");
  host.className = "pdf-export-host";
  const report = $("report-card").cloneNode(true);
  // 避免 html2pdf/html2canvas 在处理重复 id 时重新命中页面里的原卡片。
  report.removeAttribute("id");
  report.classList.remove("hide");
  report.classList.add("pdf-export-report");
  Object.assign(report.style, {
    width: "186mm",
    margin: "0",
    padding: "0",
    border: "0",
    borderRadius: "0",
    boxShadow: "none",
    background: "#fff",
  });
  report.querySelector(".report-actions")?.remove();
  splitLongPdfParagraphs(report);
  host.appendChild(report);
  document.body.appendChild(host);
  return { report, remove: () => host.remove() };
}

function textBoundary(root, absoluteOffset) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let remaining = absoluteOffset;
  let node = walker.nextNode();
  while (node) {
    if (remaining <= node.data.length) return { node, offset: remaining };
    remaining -= node.data.length;
    node = walker.nextNode();
  }
  return { node: root, offset: root.childNodes.length };
}

function paragraphCutPoints(text, targetLength = 140) {
  if (text.length <= targetLength * 1.4) return [];
  const sentenceEnds = [];
  const pattern = /[。！？；.!?](?:[”’」』】])?\s*/g;
  let match;
  while ((match = pattern.exec(text)) !== null) sentenceEnds.push(match.index + match[0].length);
  const cuts = [];
  let start = 0;
  while (text.length - start > targetLength * 1.4) {
    const preferred = sentenceEnds.filter(point => point > start + 70 && point <= start + 190);
    const cut = preferred.length ? preferred[preferred.length - 1] : start + targetLength;
    cuts.push(cut);
    start = cut;
  }
  return cuts;
}

function splitLongPdfParagraphs(report) {
  for (const paragraph of report.querySelectorAll("#report p")) {
    const text = paragraph.textContent || "";
    const cuts = paragraphCutPoints(text);
    if (!cuts.length) {
      paragraph.classList.add("pdf-paragraph-fragment");
      continue;
    }
    const group = document.createElement("div");
    group.className = "pdf-paragraph-group";
    const boundaries = [0, ...cuts, text.length];
    for (let index = 0; index < boundaries.length - 1; index++) {
      const range = document.createRange();
      const start = textBoundary(paragraph, boundaries[index]);
      const end = textBoundary(paragraph, boundaries[index + 1]);
      range.setStart(start.node, start.offset);
      range.setEnd(end.node, end.offset);
      const fragment = paragraph.cloneNode(false);
      fragment.classList.add("pdf-paragraph-fragment");
      fragment.appendChild(range.cloneContents());
      group.appendChild(fragment);
    }
    paragraph.replaceWith(group);
  }
}

$("report-download").onclick = () => showActionResult(
  $("report-download"),
  "生成中…",
  "已下载",
  async () => {
    if (!window.html2pdf || window.__html2pdf_failed) {
      throw new Error("PDF 组件加载失败，请检查网络后刷新页面。");
    }
    const exported = createPdfExportNode();
    try {
      await window.html2pdf()
        .set({
          margin: [10, 12, 12, 12],
          filename: reportPdfFilename(),
          image: { type: "jpeg", quality: 0.97 },
          html2canvas: {
            scale: 2,
            useCORS: true,
            backgroundColor: "#ffffff",
            scrollX: 0,
            scrollY: 0,
          },
          jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
          pagebreak: {
            mode: ["css", "legacy"],
            avoid: [".pdf-paragraph-fragment", "li", "table", "blockquote", "pre", "figure"],
          },
        })
        .from(exported.report)
        .save();
    } finally {
      exported.remove();
    }
  },
);

async function consumeStream(runId) {
  // EventSource 无法携带 Authorization 头 → 用 fetch 流式解析 SSE（~20 行）。
  const ctrl = new AbortController();
  streamAbort = ctrl;
  for (;;) {
    if (ctrl.signal.aborted) return;
    try {
      const resp = await fetch("/api/runs/" + runId + "/events", { headers: authHeaders(), signal: ctrl.signal });
      if (resp.status === 401) { logout(); return; }
      const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
      let buf = "";
      let sawDone = false;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += value;
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
          if (block.startsWith(":")) continue; // ping
          let event = "message", data = "{}";
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) event = line.slice(7).trim();
            else if (line.startsWith("data: ")) data = line.slice(6);
          }
          let payload; try { payload = JSON.parse(data); } catch { continue; }
          handleFrame(event, payload);
          if (event === "done") sawDone = true;
        }
      }
      if (sawDone) return true;
      if (ctrl.signal.aborted) return false;
      await new Promise((r) => setTimeout(r, 1000)); // 断线自动重连（服务端 seq 去重保证不重不漏）
    } catch (e) {
      if (ctrl.signal.aborted) return false;
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
}

let currentRunId = null;
async function renderRun(runId) {
  currentRunId = runId;
  if (streamAbort) { streamAbort.abort(); streamAbort = null; }
  lastSeq = 0;
  resetProgressView();
  $("report-card").classList.add("hide");
  $("run-cancel").disabled = false;
  show($("view-run"));
  let detail;
  try { detail = await api("/api/runs/" + runId); } catch (e) { progressLine(e.message, true); return; }
  $("run-query").textContent = detail.query;
  $("run-title").textContent = "研究：" + detail.query;
  $("run-title").title = detail.query;
  setStatus(detail.status, detail);
  if (["completed", "failed", "cancelled"].includes(detail.status)) {
    // 已终结的 run 也要回放事件重建进度（卡片/规划旁白），不能只跳报告。
    const replayedDone = await consumeStream(runId);
    if (!replayedDone) onRunDone({ status: detail.status });
    return;
  }
  consumeStream(runId); // 进行中：订阅实时流（服务端会先回放已落库段）
}

$("run-back").onclick = () => { location.hash = "#/"; };
$("run-cancel").onclick = async () => {
  $("run-cancel").disabled = true;
  try { await api("/api/runs/" + currentRunId + "/cancel", { method: "POST" }); }
  catch (e) { progressLine(e.message, true); $("run-cancel").disabled = false; }
  // 状态收敛交给 SSE 的 done 帧，不做乐观更新。
};

$("clarification-submit").onclick = async () => {
  const custom = $("clarification-other").value.trim();
  const selected = document.querySelector('input[name="clarification-choice"]:checked');
  const answer = custom || (selected && selected.value) || "";
  if (!answer) { $("clarification-error").textContent = "请选择或填写一个答案。"; return; }
  $("clarification-submit").disabled = true;
  $("clarification-error").textContent = "";
  try {
    const resumed = await api("/api/runs/" + currentRunId + "/resume", {
      method: "POST", body: JSON.stringify({ answer }),
    });
    $("clarification-card").classList.add("hide");
    // API 只完成持久化入队；真正开始执行必须等 Worker claim 后的 running 事件。
    setStatus(resumed.status || "queued");
    consumeStream(currentRunId);
  } catch (e) {
    $("clarification-error").textContent = e.message;
    $("clarification-submit").disabled = false;
  }
};

/* 正文 [来源N] → 面板条目：不走 hash 导航（会与 #/ 路由打架导致切视图），
   改为 preventDefault + JS 滚动 + 手动高亮。 */
$("report").addEventListener("click", (e) => {
  const link = e.target.closest && e.target.closest('a[href^="#src-"]');
  if (!link) return;
  e.preventDefault();
  const target = document.getElementById(link.getAttribute("href").slice(1));
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.remove("flash");
  void target.offsetWidth;  // 重启动画：连点同一来源也能再次高亮
  target.classList.add("flash");
});

/* ---------- 启动 ---------- */
if (!localStorage.getItem(tokenKey)) setAuthMode("login");
showView();
