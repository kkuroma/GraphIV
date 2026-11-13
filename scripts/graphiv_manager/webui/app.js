// The manager page: the snapshot on the left, the dashboard and the stages on the right
const PALETTES = ["haruhana", "natsumikan", "akiba", "fuyuyuki"];
const SEEDS = { haruhana: "#F07898", natsumikan: "#E08030", akiba: "#C84040", fuyuyuki: "#78B4E8" };
const INTENTS = ["background", "uses", "compares", "critiques", "extends", "motivates", "unlabeled"];
const PAPER_STAGES = ["added", "pdf", "html", "embedded", "scored"];
const TABS = ["dashboard", "download", "history", "config"];
const DASH = [["overview", "Overview"], ["papers", "Papers"], ["tree", "Topic tree"], ["citations", "Citations"], ["storage", "Storage"]];
const POLL = 3000;

let S = null;
let ST = {};
let statsSeen = "";
let where = { tab: "dashboard", sub: "overview", table: "papers", stage: "papers.0", split: "" };
const speeds = {};
const dryText = {};
let showing = { key: "", mode: "", job: "", offset: 0, text: "" };

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const n = (v) => (v == null ? "–" : Number(v).toLocaleString());
const $ = (id) => document.getElementById(id);

function bytes(b) {
  if (b == null) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(b >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

function dur(s) {
  if (s == null || !isFinite(s)) return "–";
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  const h = Math.floor(s / 3600);
  return h < 48 ? `${h}h ${Math.floor((s % 3600) / 60)}m` : `${Math.floor(h / 24)}d ${h % 24}h`;
}

function ago(t) {
  if (!t) return "never";
  const s = Date.now() / 1000 - t;
  return s < 60 ? "just now" : `${dur(s)} ago`;
}

const when = (t) => (t ? new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–");
const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(a / b < 0.1 ? 2 : 1)}%` : "–");

// -- theme -----------------------------------------------------------------------------------

function apply(palette, mode) {
  document.documentElement.dataset.palette = palette;
  document.documentElement.dataset.mode = mode;
  try {
    localStorage.setItem("graphiv.palette", palette);
    localStorage.setItem("graphiv.mode", mode);
  } catch (e) { /* a private window forgets, which costs nothing */ }
  for (const b of document.querySelectorAll(".swatch")) b.setAttribute("aria-pressed", String(b.dataset.palette === palette));
}

function theme() {
  let palette = "haruhana";
  let mode = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  try {
    palette = localStorage.getItem("graphiv.palette") || palette;
    mode = localStorage.getItem("graphiv.mode") || mode;
  } catch (e) { /* as above */ }
  // a link can carry its own theme, which is how a screenshot picks one
  const asked = new URLSearchParams(location.search);
  if (PALETTES.includes(asked.get("palette"))) palette = asked.get("palette");
  if (["light", "dark"].includes(asked.get("mode"))) mode = asked.get("mode");
  for (const name of PALETTES) {
    const b = document.createElement("button");
    b.className = "swatch";
    b.dataset.palette = name;
    b.title = name;
    b.style.background = SEEDS[name];
    b.onclick = () => apply(name, document.documentElement.dataset.mode);
    $("swatches").append(b);
  }
  $("mode").onclick = () => apply(document.documentElement.dataset.palette,
    document.documentElement.dataset.mode === "dark" ? "light" : "dark");
  apply(palette, mode);
}

// -- the server ------------------------------------------------------------------------------

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  const got = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(got.error || `${r.status}`);
  return got;
}

async function act(path, body, then) {
  try {
    const got = await api(path, body);
    if (then) then(got);
    await poll();
  } catch (e) {
    alert(e.message);
  }
}

// -- hover -----------------------------------------------------------------------------------

document.addEventListener("mousemove", (e) => {
  const tip = $("tip");
  const el = e.target.closest ? e.target.closest("[data-tip]") : null;
  if (!el) { tip.style.display = "none"; return; }
  tip.innerHTML = el.dataset.tip;
  tip.style.display = "block";
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  let x = e.clientX + 14;
  let y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 14;
  if (y + h > innerHeight - 8) y = e.clientY - h - 14;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
});

// -- the status lights -----------------------------------------------------------------------

function leds() {
  const h = S.health || {};
  const one = (label, ok, note) => `<span class="ledchip" data-tip="${esc(note)}"><i class="led ${ok ? "ok" : "bad"}"></i>${label}</span>`;
  const g = h.gpu || {};
  const disks = Object.entries(h.disk || {});
  const low = disks.filter(([, v]) => !v.ok);
  $("leds").innerHTML = [
    one("Database", (h.store || {}).ok, (h.store || {}).note || ""),
    one("Language model", (h.llm || {}).ok, (h.llm || {}).note || ""),
    one("Embeddings", (h.embed || {}).ok, (h.embed || {}).note || ""),
    one(g.ok ? `GPU ${g.load}%` : "GPU", g.ok, g.ok ? `${esc(g.name)}<br>${n(g.used)} of ${n(g.total)} MiB in use` : g.note || ""),
    one("Disks", !low.length, disks.map(([k, v]) => `${esc(k)}: ${v.free != null ? bytes(v.free) + " free of " + bytes(v.total) : esc(v.note)}`).join("<br>")),
  ].join("");
  const q = S.queues || {};
  $("counted").textContent = S.counting ? "counting queues…" : q.taken ? `queues counted ${ago(q.taken)}` : "queues not counted yet";
}

// -- the rail: the source, the snapshot in use, what is staged, and what is running --------

function running() {
  return (S.jobs || []).find((j) => j.status === "running" || j.status === "stopping");
}

// speed is taken between polls rather than since the step began, so a second phase reads true
function speed(key, done) {
  const now = Date.now() / 1000;
  const was = speeds[key];
  if (!was || done < was.done) { speeds[key] = { done, t: now, rate: null }; return null; }
  if (now - was.t < 2) return was.rate;
  const fresh = (done - was.done) / (now - was.t);
  const rate = was.rate == null ? fresh : 0.6 * was.rate + 0.4 * fresh;
  speeds[key] = { done, t: now, rate };
  return rate;
}

function stepProgress(job, stageKey) {
  const step = job.steps[job.current] || {};
  const st = (S.stages || []).find((s) => s.key === (stageKey || step.stage)) || {};
  const [done, total0] = job.progress || [null, null];
  const total = total0 || (done != null ? st.waiting : null);
  const spent = step.started ? S.now - step.started : 0;
  const rate = done != null ? speed(`${job.id}.${job.current}.${job.phase || ""}`, done) : null;
  const eta = rate && total ? (total - done) / rate : null;
  const unit = job.phase === "fetch" || job.phase === "unzip" ? "bytes" : st.unit || "rows";
  return { step, done, total, spent, rate, eta, unit };
}

function amount(v, unit) {
  return unit === "bytes" ? bytes(v) : n(v);
}

function pace(rate, unit) {
  if (!rate) return "";
  if (unit === "bytes") return `${bytes(rate)}/s`;
  return `${rate >= 10 ? n(Math.round(rate)) : rate.toFixed(2)}/s`;
}

function progressHtml(job, big) {
  const p = stepProgress(job);
  const width = p.done != null && p.total ? Math.min(100, (100 * p.done) / p.total) : null;
  const bar = `<div class="bar live${big ? " big" : ""}${width == null ? " busy" : ""}"><i style="width:${width ?? 0}%"></i>${big && width != null ? `<b>${width.toFixed(1)}%</b>` : ""}</div>`;
  const got = `<b>${p.done != null ? amount(p.done, p.unit) : "no counter yet"}</b>${p.total ? " of " + amount(p.total, p.unit) : ""}`;
  const left = p.eta != null ? `<b>${dur(p.eta)}</b> left` : `${dur(p.spent)} in`;
  if (!big) {
    // the rail is narrow, so the compact form is two short lines
    return `${bar}<div class="barline"><span>${got}</span><span><b>${width != null ? width.toFixed(1) + "%" : ""}</b></span></div>
      <div class="barline"><span>${p.rate ? `<b>${pace(p.rate, p.unit)}</b>` : ""}</span><span>${left}</span></div>`;
  }
  return `${bar}<div class="barline"><span>${got}</span>
    <span>${p.rate ? `<b>${pace(p.rate, p.unit)}</b> · ` : ""}${dur(p.spent)} in${p.eta != null ? ` · <b>${dur(p.eta)}</b> left` : ""}</span></div>`;
}

const PHASES = { fetch: "Downloading archive.zip", unzip: "Unzipping to arxiv-new.json", stage0: "Comparing it with the store" };

function fetchBox() {
  const job = (S.jobs || []).find((j) => ["queued", "waiting", "running", "stopping"].includes(j.status) && j.steps.some((s) => s.kind === "fetch"));
  if (!job) return "";
  const live = job.status === "running" || job.status === "stopping";
  const phase = live ? PHASES[job.phase] || (job.steps[job.current] || {}).label : job.reason || "waiting to start";
  return `<div class="box fetching"><div class="box-t"><i class="led ${live ? "run" : "warn"}"></i>${esc(job.title)}<span class="tech">step ${job.current + 1} of ${job.steps.length}</span></div>
    <div class="phase">${esc(phase)}</div>${live ? progressHtml(job) : ""}
    <div class="btns"><button class="btn sm" data-open="${job.id}">Open</button><button class="btn sm danger" data-cancel="${job.id}">${job.status === "stopping" ? "Stopping…" : "Stop"}</button></div>
    <div class="tech" style="margin-top:6px">A stopped download resumes where it stopped.</div></div>`;
}

// a download that was stopped, which the fetch button resumes while Kaggle serves the same file
function partBox(part) {
  const share = part.total ? part.size / part.total : 0;
  return `<div class="box fetching"><div class="box-t"><i class="led warn"></i>${part.unzip ? "Downloaded, not unzipped" : "Download stopped"}<span class="tech">arxiv-new.zip</span></div>
    <div class="phase">${part.unzip ? "The whole archive is here and unzips on the next fetch." : `${bytes(part.size)}${part.total ? ` of ${bytes(part.total)}` : ""}, which the next fetch resumes.`}</div>
    ${part.total ? `<div class="bar"><i style="width:${(100 * share).toFixed(1)}%"></i></div><div class="barline"><span>${pct(part.size, part.total)}</span><span>starts over if Kaggle has published since</span></div>` : ""}
    <div class="btns"><button class="btn sm danger" id="droppart">Discard</button></div></div>`;
}

function rail() {
  const shot = S.snapshot || {};
  const cur = shot.current || {};
  const staged = shot.staged;
  const check = S.check || {};
  const dbNewest = (S.queues || {}).newest;
  const agrees = !staged && cur.newest && dbNewest && cur.newest <= dbNewest;
  const checking = (S.jobs || []).find((j) => ["queued", "waiting", "running"].includes(j.status) && j.title.startsWith("Check the"));
  const remote = S.remote || {};
  const published = remote.published ? Date.parse(remote.published) / 1000 : null;
  const newer = published && published > (cur.modified || 0) && !(staged && staged.modified > published);
  const busy = fetchBox();
  const part = shot.partial;
  let html = `<div class="rh">Source<span class="tech">${remote.taken ? "asked " + ago(remote.taken) : "not asked yet"}</span></div>
    <div class="box source${newer ? " newer" : ""}"><div class="box-t"><i class="led ${remote.version ? (newer ? "warn" : "ok") : "bad"}"></i>Kaggle<span class="tech mono">Cornell-University/arxiv</span></div>
      ${remote.version ? `<div class="ver"><b>v${esc(remote.version)}</b><span>published ${when(published)}</span></div>
      <div class="kv"><span>Unzipped</span><span>${bytes(remote.size)}</span><span>In use</span><span>${S.source && S.source.version ? "v" + esc(S.source.version) : "written " + when(cur.modified)}</span></div>
      <div class="verdict">${newer ? "Newer than the file in use." : staged && published && staged.modified > published ? "Already fetched and staged below." : "The file in use is already this version or newer."}</div>`
        : `<div class="tech" style="margin-top:6px">Kaggle's metadata api has not answered, so the version it serves is unknown.</div>`}
      <button class="btn ${newer || part ? "primary" : ""} wide" id="fetch"${busy ? " disabled" : ""}>${busy ? "Fetching…" : part ? (part.unzip ? "Unzip the download" : "Resume the download") : "Fetch new source json"}</button></div>
    ${busy || (part ? partBox(part) : "")}
    <div class="rh">In use</div>
    <div class="box"><div class="box-t"><i class="led ${agrees ? "ok" : staged ? "warn" : "idle"}"></i>arxiv.json<span class="tech">in use</span></div>
      <div class="kv"><span>Size</span><span>${bytes(cur.size)}</span>
        <span>Newest paper in file</span><span>${esc(cur.newest || "–")}</span>
        <span>Newest paper in store</span><span>${esc(dbNewest || "–")}</span>
        <span>Replaced</span><span>${when(cur.modified)}</span></div>
      ${(S.checked || {}).new != null ? `<div class="kv"><span>Checked against the store</span><span>${n(S.checked.new)} new, ${n(S.checked.reset)} revised</span></div>` : ""}
      <div class="tech" style="margin-top:6px">${agrees ? "The store already holds everything in this file." : staged ? "A newer snapshot is staged below." : "Fetch a newer snapshot to bring the store forward."}</div></div>`;
  if (staged) {
    const found = (check.new || 0) + (check.reset || 0);
    const forecast = S.forecast || [];
    const total = forecast.reduce((a, f) => a + (f.seconds || 0), 0);
    html += `<div class="rh">Staged</div><div class="box staged">
      <div class="box-t"><i class="led warn"></i>arxiv-new.json<span class="tech">${when(staged.modified)}</span></div>
      <div class="kv"><span>Size</span><span>${bytes(staged.size)}</span><span>Newest paper</span><span>${esc(staged.newest)}</span>
        ${check.new != null ? `<span>New papers</span><span>${n(check.new)}</span><span>New versions</span><span>${n(check.reset)}</span>
        <span>Files to remove</span><span>${n(check.doomed)}</span><span>Citations to drop</span><span>${n(check.edges)}</span>` : ""}</div>
      ${check.new == null ? `<div style="margin-top:8px">${checking ? `<div class="bar busy"><i></i></div><div class="barline"><span>Comparing it with the store</span><span>${esc(checking.status)}</span></div>`
        : `<button class="btn sm" id="recheck">Compare it with the store</button>`}</div>` : ""}
      ${staged.newest && cur.newest && staged.newest < cur.newest ? `<div class="tech" style="color:var(--red);margin-top:6px">This file ends at ${esc(staged.newest)}, before the one in use, which ends at ${esc(cur.newest)}.</div>` : ""}
      ${check.new === 0 && !check.reset ? `<div class="tech" style="margin-top:6px">Nothing in this file is new to the store, so a full run would only drain what is already queued.</div>` : ""}
      ${check.new != null ? `<div class="rh" style="margin-top:12px">What a full run does<span class="tech">${total ? "about " + dur(total) + " measured" : ""}</span></div>
        <table class="fc">${forecast.map((f) => `<tr class="${f.rows ? "hot" : ""}"><td>${esc(f.label)}</td><td>${n(f.rows)}</td><td>${f.seconds ? dur(f.seconds) : "–"}</td></tr>`).join("")}</table>
        <div class="tech" style="margin-top:6px">Rows are what each stage works through once this file is in use. Times use rates measured here, and a dash means none has been measured.</div>` : ""}
      <div class="btns"><button class="btn primary" id="runall"${check.new == null ? " disabled" : ""}>Run everything</button>
        <button class="btn" id="dryall">Dry run</button><button class="btn" id="commit">Commit only</button>
        <button class="btn danger" id="discard">Discard</button></div></div>`;
  }
  const live = running();
  html += `<div class="rh">Running now</div>`;
  if (live && live.steps.some((s) => s.kind === "fetch")) html += `<div class="tech">The fetch above.</div>`;
  else if (live) {
    html += `<div class="box"><div class="box-t"><i class="led run"></i>${esc(live.title)}<span class="tech">step ${live.current + 1} of ${live.steps.length}</span></div>
      <div class="tech" style="margin:4px 0 8px">${esc((live.steps[live.current] || {}).label || "")}</div>${progressHtml(live)}
      <div class="btns"><button class="btn sm" data-open="${live.id}">Open</button><button class="btn sm danger" data-cancel="${live.id}">${live.status === "stopping" ? "Stopping…" : "Stop"}</button></div></div>`;
  } else html += `<div class="tech">Nothing is running.</div>`;
  const next = (S.jobs || []).filter((j) => ["queued", "scheduled", "waiting"].includes(j.status)).reverse();
  if (next.length) {
    html += `<div class="rh">Up next</div><div class="box">${next.map((j) => `<div class="jobline"><i class="led ${j.status === "waiting" ? "warn" : "idle"}"></i>
      <span class="grow" data-tip="${esc(j.reason || j.status)}">${esc(j.title)}</span><button class="btn sm" data-cancel="${j.id}">Remove</button></div>`).join("")}</div>`;
  }
  $("rail").innerHTML = html;
  wireRail();
}

function wireRail() {
  const on = (id, fn) => { const b = $(id); if (b) b.onclick = fn; };
  on("fetch", () => act("/api/fetch", {}));
  on("recheck", () => act("/api/check", {}));
  on("runall", () => confirm("Put the staged snapshot in place and run every stage in order?") && act("/api/run", { all: true }, openRunning));
  on("dryall", () => { where = { ...where, tab: "download" }; dry({ all: true }, where.stage); });
  on("commit", () => confirm("Replace arxiv.json with the staged file now?") && act("/api/snapshot/commit", {}));
  on("droppart", () => confirm("Delete the partial download?") && act("/api/snapshot/discard", {}));
  on("discard", () => confirm("Delete the staged arxiv-new.json and any partial download?") && act("/api/snapshot/discard", {}));
  for (const b of document.querySelectorAll("[data-cancel]")) b.onclick = () => act(`/api/jobs/${b.dataset.cancel}/cancel`, {});
  for (const b of document.querySelectorAll("[data-open]")) b.onclick = () => openJob(b.dataset.open);
}

// -- routing ---------------------------------------------------------------------------------

function go(next) {
  where = { ...where, ...next };
  const path = where.tab === "dashboard" ? `#/dashboard/${where.sub}`
    : where.tab === "download" ? `#/download/${where.table}/${where.stage}`
    : where.tab === "config" ? "#/config"
    : "#/history";
  if (location.hash !== path) history.replaceState(null, "", path);
  draw(true);
}

function readHash() {
  const [, tab, a, b] = location.hash.split("/");
  if (tab === "dashboard") where = { ...where, tab, sub: a || "overview" };
  else if (tab === "download") where = { ...where, tab, table: a || "papers", stage: b || `${a || "papers"}.0` };
  else if (tab === "history" || tab === "config") where = { ...where, tab };
}

function openJob(id) {
  const job = (S.jobs || []).find((j) => j.id === id);
  const step = job && (job.steps[job.current] || job.steps[job.steps.length - 1]);
  const st = step && (S.stages || []).find((s) => s.key === step.stage);
  if (st) go({ tab: "download", table: st.table, stage: st.key });
  else go({ tab: "history" });
}

function openRunning(job) {
  const st = job && job.steps && (S.stages || []).find((s) => s.key === (job.steps.find((x) => x.stage) || {}).stage);
  if (st) go({ tab: "download", table: st.table, stage: st.key });
}

function draw(full) {
  for (const b of $("tabs").querySelectorAll("button")) b.classList.toggle("active", b.dataset.t === where.tab);
  for (const t of TABS) $(`p-${t}`).classList.toggle("active", t === where.tab);
  if (where.tab === "dashboard") dashboard(full);
  if (where.tab === "download") download(full);
  if (where.tab === "history") historyTab();
  if (where.tab === "config" && full) configTab();
}

// -- charts ----------------------------------------------------------------------------------

// every chart draws from its domain's two hues, which the stylesheet sets per palette and mode
// two palettes give secondary and yellow one value, so motivates is mixed to stay apart from uses
const INTENT_COLOR = {
  background: "--overlay1", uses: "--secondary", compares: "--sky", critiques: "--red",
  extends: "--green", motivates: "color-mix(in oklab, var(--yellow) 50%, var(--red))", unlabeled: "--surface2",
};
const hue = (c) => (c.startsWith("--") ? `var(${c})` : c);
const STAGE_COLOR = ["--overlay1", "--sky", "--secondary", "--yellow", "--green"];
const ICONS = {
  papers: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/>',
  cite: '<path d="M10 7H6v5h4v-5zM18 7h-4v5h4v-5z"/><path d="M6 12c0 3-1 5-3 6M14 12c0 3-1 5-3 6"/>',
  tree: '<circle cx="12" cy="5" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="18" r="2"/><path d="M12 7v5M12 12H6v4M12 12h6v4"/>',
  disk: '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/>',
  spark: '<path d="M13 3 5 14h6l-1 7 8-11h-6z"/>',
  check: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
};
const icon = (name) => `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;

// a peak label has room for about five characters
function short(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e4) return `${Math.round(v / 1e3)}k`;
  return n(v);
}
const rows = (q) => (ST[q] ? ST[q].rows : []);
const none = `<div class="pending">Nothing counted yet.</div>`;

// sizes spread over orders of magnitude, so they are binned on a log scale
function logBins(values, bins = 14) {
  if (!values.length) return [];
  const lo = Math.max(1, Math.min(...values));
  const hi = Math.max(...values);
  const a = Math.log(lo);
  const b = Math.log(hi + 1);
  const out = Array.from({ length: bins }, (_, i) => [Math.round(Math.exp(a + ((b - a) * i) / bins)), 0]);
  for (const v of values) out[Math.min(bins - 1, Math.floor(((Math.log(Math.max(1, v)) - a) / (b - a)) * bins))][1]++;
  return out;
}

// a distribution: each bar takes its hue from where it sits on the axis
function cols(data, o = {}) {
  if (!data.length) return none;
  const f = o.sqrt ? Math.sqrt : (x) => x;
  const max = f(Math.max(...data.map((r) => r[1]), 1));
  const lab = o.label || ((x) => x);
  const peak = data.reduce((b, r, i) => (r[1] > data[b][1] ? i : b), 0);
  const last = Math.max(1, data.length - 1);
  const bars = data.map(([k, v], i) => {
    const h = Math.max(0.6, (100 * f(v)) / max);
    return `<div class="${i === peak ? "peak" : ""}" style="--h:${h}%;--t:${Math.round((100 * i) / last)}%" data-tip="${esc(lab(k))}<br><b>${n(v)}</b>${o.total ? " · " + pct(v, o.total) : ""}">
      ${i === peak ? `<em>${short(v)}</em>` : ""}<i></i></div>`;
  }).join("");
  const ticks = o.ticks || [data[0][0], data[Math.floor(data.length / 2)][0], data[data.length - 1][0]].map(lab);
  return `<div class="cols${o.tall ? " tall" : ""}${o.short ? " short" : ""}">${bars}</div><div class="colx">${ticks.map((t) => `<span>${esc(t)}</span>`).join("")}</div>`;
}

// a series over time: a filled line, with a hit column per point for the hover
function area(data, o = {}) {
  if (data.length < 2) return cols(data, o);
  const max = Math.max(...data.map((r) => r[1]), 1);
  const w = data.length;
  const pts = data.map(([, v], i) => [i + 0.5, 100 - (96 * v) / max]);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join("");
  const fill = `${line}L${(w - 0.5).toFixed(2)},100L0.5,100Z`;
  const lab = o.label || ((x) => x);
  const peak = data.reduce((b, r, i) => (r[1] > data[b][1] ? i : b), 0);
  const hits = data.map(([k, v], i) => `<div style="--y:${100 - pts[i][1]}%" data-tip="${esc(lab(k))}<br><b>${n(v)}</b>${o.total ? " · " + pct(v, o.total) : ""}"><i></i></div>`).join("");
  const ticks = o.ticks || [data[0][0], data[Math.floor(w / 2)][0], data[w - 1][0]].map(lab);
  return `<div class="area${o.tall ? " tall" : ""}${o.short ? " short" : ""}">
    <svg viewBox="0 0 ${w} 100" preserveAspectRatio="none" aria-hidden="true">
      <path d="${fill}" class="fill"/>
      <path d="${line}" fill="none" class="stroke" vector-effect="non-scaling-stroke"/>
    </svg>
    <span class="pk" style="left:${((peak + 0.5) / w) * 100}%;bottom:${100 - pts[peak][1]}%"><em>${short(data[peak][1])}</em></span>
    <div class="hits">${hits}</div></div>
    <div class="colx">${ticks.map((t) => `<span>${esc(t)}</span>`).join("")}</div>`;
}

// ranked amounts: the hue walks down the ramp with the rank
function hbars(data, o = {}) {
  if (!data.length) return none;
  const max = Math.max(...data.map((r) => r[1]), 1);
  const total = o.total || data.reduce((a, r) => a + r[1], 0);
  const fmt = o.fmt || n;
  const last = Math.max(1, data.length - 1);
  const row = ([k, v, color], i) => `
    <span class="rank${i < 3 ? " podium" : ""}">${i + 1}</span>
    <span class="lab" title="${esc(k)}">${esc(k)}</span>
    <span class="trk" data-tip="${esc(k)}<br><b>${fmt(v)}</b> · ${pct(v, total)}"><i style="width:${(100 * v) / max}%;--t:${Math.round((100 * i) / last)}%${color ? `;--c:${hue(color)}` : ""}"></i></span>
    <span class="val">${fmt(v)}</span><span class="share">${pct(v, total)}</span>`;
  if (!o.split) return `<div class="hb">${data.map(row).join("")}</div>`;
  // a long ranking reads down one column and continues in the next
  const half = Math.ceil(data.length / 2);
  return `<div class="hb-split"><div class="hb">${data.slice(0, half).map(row).join("")}</div>
    <div class="hb">${data.slice(half).map((r, i) => row(r, i + half)).join("")}</div></div>`;
}

// parts of a whole: a ring with the headline in the middle and a legend that carries the numbers
function donut(parts, center, o = {}) {
  const total = parts.reduce((a, p) => a + p[1], 0);
  if (!total) return none;
  let at = 0;
  const gap = parts.filter((p) => p[1]).length > 1 ? 0.35 : 0;
  const segs = parts.filter((p) => p[1]).map(([, v, c]) => {
    const from = at;
    at += (100 * v) / total;
    return `${hue(c)} ${from.toFixed(3)}% ${Math.max(from, at - gap).toFixed(3)}%, transparent ${Math.max(from, at - gap).toFixed(3)}% ${at.toFixed(3)}%`;
  }).join(", ");
  const max = Math.max(...parts.map((p) => p[1]));
  return `<div class="donut-wrap${o.small ? " sm" : ""}"><div class="donut" style="--seg:conic-gradient(from -90deg, ${segs})"><div class="donut-c"><b>${center.v}</b><span>${center.k}</span></div></div>
    <div class="dleg">${parts.map(([label, v, c]) => `<div class="dl${v ? "" : " zero"}" data-tip="${esc(label)}<br><b>${n(v)}</b> · ${pct(v, total)}">
      <i style="background:${hue(c)}"></i><span class="dl-k">${esc(label)}</span><span class="dl-v">${o.fmt ? o.fmt(v) : n(v)}</span><span class="dl-p">${pct(v, total)}</span>
      <span class="dl-bar"><s style="width:${(100 * v) / max}%;background:${hue(c)}"></s></span></div>`).join("")}</div></div>`;
}

// a percentage as a small ring, for a tile
function ring(share, color) {
  const p = Math.max(0, Math.min(100, share * 100));
  return `<div class="ring" style="--p:${p}%${color ? `;--c:${hue(color)}` : ""}"><span>${p >= 99.95 ? "100" : p.toFixed(p < 10 ? 1 : 0)}<small>%</small></span></div>`;
}

// a trend for a tile: the shape only, since the chart below carries the numbers
function spark(data) {
  if (data.length < 2) return "";
  const max = Math.max(...data.map((r) => r[1]), 1);
  const pts = data.map(([, v], i) => `${((i / (data.length - 1)) * 100).toFixed(2)},${(34 - (30 * v) / max).toFixed(2)}`);
  return `<svg class="spark" viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0,36L${pts.join("L")}L100,36Z" class="fill"/><path d="M${pts.join("L")}" fill="none" class="stroke" vector-effect="non-scaling-stroke"/></svg>`;
}

function chart(title, body, o = {}) {
  const took = o.q && ST[o.q] ? `${ST[o.q].seconds}s · ${ago(ST[o.q].taken)}` : "";
  return `<section class="chart ${o.w || ""} ${o.dom || ""}">
    <header class="chart-h">${o.icon ? icon(o.icon) : ""}<h3>${title}</h3><span class="tech" title="how long the query took, and when it ran">${took}</span></header>
    ${o.sub ? `<p class="chart-s">${o.sub}</p>` : ""}${body}</section>`;
}

function tile(o) {
  const v = String(o.v ?? "–");
  return `<div class="tile ${o.dom || ""}${o.hot ? " hot" : ""}">
    <div class="tile-top">${o.icon ? icon(o.icon) : ""}<span class="tile-k">${o.k}</span></div>
    <div class="tile-body"><div><div class="tile-v${v.length > 11 ? " long" : ""}">${v}</div><div class="tile-s">${o.s || "&nbsp;"}</div></div>
      ${o.ring != null ? ring(o.ring, o.color) : ""}</div>${o.spark ? spark(o.spark) : ""}</div>`;
}

// -- the dashboard ---------------------------------------------------------------------------

function dashboard(full) {
  const host = $("p-dashboard");
  const seen = JSON.stringify(S.stats_taken || {}) + where.sub + S.summing;
  if (!full && seen === statsSeen) return;
  statsSeen = seen;
  const oldest = Math.min(...Object.values(S.stats_taken || {}).filter(Boolean));
  host.innerHTML = `<div class="subbar"><div class="seg">${DASH.map(([k, l]) => `<button data-d="${k}" class="${where.sub === k ? "on" : ""} d-${k}"><i></i>${l}</button>`).join("")}</div>
    <span class="spacer"></span><span class="tech">${S.summing ? "recomputing, charts update as each query lands" : isFinite(oldest) ? `oldest number from ${ago(oldest)}` : "not computed yet"}</span>
    <button class="btn sm" id="restat"${S.summing ? " disabled" : ""}>${S.summing ? "Recomputing…" : "Recompute"}</button></div><div id="dash"></div>`;
  for (const b of host.querySelectorAll("[data-d]")) b.onclick = () => go({ sub: b.dataset.d });
  $("restat").onclick = () => act("/api/stats/refresh", {});
  const body = { overview, papers: papersDash, tree: treeDash, citations: citeDash, storage: storageDash }[where.sub] || overview;
  $("dash").innerHTML = Object.keys(ST).length ? body() : `<div class="pending">The dashboard numbers are being computed for the first time. The citation charts read a 54 GB table and take a few minutes.</div>`;
  if (where.sub === "tree" && Object.keys(ST).length) mountTree();
}

function intentTotals() {
  const by = {};
  const stage = { 0: 0, 1: 0 };
  for (const [i, s, v] of rows("c_intent")) { by[i] = (by[i] || 0) + v; stage[s] = (stage[s] || 0) + v; }
  return { by, stage };
}

function overview() {
  const [total, first, last] = rows("p_totals")[0] || [];
  const stage = Object.fromEntries(rows("p_stage"));
  const embedded = (stage[3] || 0) + (stage[4] || 0);
  const ct = Object.fromEntries(rows("c_totals"));
  const { by, stage: cs } = intentTotals();
  const names = Object.fromEntries(rows("t_names"));
  const pub = rows("p_published")[0] || [];
  const db = (rows("s_database")[0] || [])[0];
  const labeledShare = ct.resolved ? cs[1] / ct.resolved : 0;
  return `<div class="tiles hero">
      ${tile({ k: "Papers", v: n(total), s: `${esc(first || "")} to ${esc(last || "")}`, icon: "papers", dom: "dom-papers", hot: true, spark: rows("p_month") })}
      ${tile({ k: "Citations", v: short(ct.rows || 0), s: `${n(ct.resolved)} resolved to a paper here`, icon: "cite", dom: "dom-cite", hot: true, ring: ct.rows ? ct.resolved / ct.rows : 0 })}
      ${tile({ k: "Embedded", v: n(embedded), s: "papers with a vector", icon: "layers", dom: "dom-papers", ring: total ? embedded / total : 0, color: "--sky" })}
      ${tile({ k: "Labeled", v: n(cs[1]), s: `${n(cs[0])} still waiting`, icon: "check", dom: "dom-cite", ring: labeledShare, color: "--green" })}
    </div><div class="tiles">
      ${tile({ k: "Scored for reviews", v: n(stage[4]), s: `${n(pub[2])} likely reviews`, icon: "spark", dom: "dom-papers" })}
      ${tile({ k: "Topics", v: `${n(names[0])} · ${n(names[1])} · ${n(names[2])}`, s: "grandparents, parents, leaves", icon: "tree", dom: "dom-tree" })}
      ${tile({ k: "Database", v: bytes(db), s: "on disk, indexes included", icon: "disk", dom: "dom-store" })}
      ${tile({ k: "Newest paper", v: esc(last || "–"), s: "in the store", icon: "clock", dom: "dom-over" })}
    </div><div class="charts">
      ${chart("Papers by stage", donut(PAPER_STAGES.map((l, i) => [l, stage[i] || 0, STAGE_COLOR[i]]), { v: pct(stage[4] || 0, total), k: "fully scored" }), { q: "p_stage", dom: "dom-papers", icon: "layers", sub: "Each paper moves through these in order, and a delta leaves the new ones at added." })}
      ${chart("Resolved citations by intent", donut(INTENTS.filter((i) => i !== "unlabeled" || by[i]).map((i) => [i, by[i] || 0, INTENT_COLOR[i]]), { v: short(ct.resolved || 0), k: "resolved" }), { q: "c_intent", dom: "dom-cite", icon: "cite", sub: "Why one paper cites another, as the labeler read the citing sentence." })}
      ${chart("Papers per year", area(rows("p_year").filter((r) => r[0] >= 1991), { total }), { q: "p_year", dom: "dom-papers", icon: "papers" })}
      ${chart("Papers per month, last three years", area(rows("p_month"), { total }), { q: "p_month", dom: "dom-over", icon: "clock" })}
    </div>`;
}

function papersDash() {
  const [total] = rows("p_totals")[0] || [];
  const pub = rows("p_published")[0] || [];
  const par = Object.fromEntries(rows("p_parsable").map(([k, v]) => [String(k), v]));
  const parsed = (par.true || 0) + (par.false || 0);
  const survey = rows("p_survey");
  const surveyTotal = survey.reduce((a, r) => a + r[1], 0);
  return `<div class="tiles dense">
      ${tile({ k: "Papers", v: n(total), s: "rows in papers", icon: "papers", dom: "dom-papers", hot: true })}
      ${tile({ k: "DOI", v: n(pub[0]), s: "published", icon: "check", dom: "dom-papers", ring: total ? pub[0] / total : 0, color: "--sky" })}
      ${tile({ k: "Journal ref", v: n(pub[1]), s: "a venue on record", icon: "check", dom: "dom-papers", ring: total ? pub[1] / total : 0, color: "--secondary" })}
      ${tile({ k: "Likely reviews", v: n(pub[2]), s: "scored 0.5 or more", icon: "spark", dom: "dom-papers" })}
      ${tile({ k: "References", v: n(par.true), s: `${n(par.false)} had none`, icon: "cite", dom: "dom-cite", ring: parsed ? (par.true || 0) / parsed : 0, color: "--green" })}
      ${tile({ k: "Never parsed", v: n(par.null), s: "for citations stage 0", icon: "clock", dom: "dom-over" })}
    </div><div class="charts">
      ${chart("Papers per year", area(rows("p_year"), { total, short: true }), { w: "w8", q: "p_year", dom: "dom-papers", icon: "papers", sub: "By the year of first submission." })}
      ${chart("Survey probability", cols(survey, { total: surveyTotal, short: true, label: (b) => `${((b - 1) / 20).toFixed(2)} to ${(b / 20).toFixed(2)}`, ticks: ["0", "0.5", "1"], sqrt: true }), { w: "w4", q: "p_survey", dom: "dom-cite", icon: "spark", sub: "The review classifier's score, square root scale." })}
      ${chart("Archives", hbars(rows("p_archive").slice(0, 15), { total }), { w: "w4", q: "p_archive", dom: "dom-papers", icon: "layers", sub: "The primary category before the dot." })}
      ${chart("Primary categories, top 30", hbars(rows("p_category"), { total, split: true }), { w: "w8", q: "p_category", dom: "dom-over", icon: "layers", sub: "&nbsp;" })}
      ${chart("Bibliography", donut([["with references", par.true || 0, "--green"], ["none found", par.false || 0, "--yellow"], ["not parsed yet", par.null || 0, "--overlay1"]], { v: pct(par.true || 0, total), k: "have references" }, { small: true }), { w: "w4", q: "p_parsable", dom: "dom-cite", icon: "cite" })}
      ${chart("Authors per paper", cols(rows("p_authors"), { total, short: true, label: (k) => (k >= 30 ? "30 or more" : `${k} authors`), ticks: ["0", "15", "30+"] }), { w: "w4", q: "p_authors", dom: "dom-tree", icon: "papers" })}
      ${chart("Latest version", cols(rows("p_versions").filter((r) => r[0] != null), { total, short: true, sqrt: true, label: (v) => (v >= 10 ? "v10 or later" : `v${v}`), ticks: ["v1", "v5", "v10+"] }), { w: "w4", q: "p_versions", dom: "dom-store", icon: "layers" })}
    </div>`;
}

function treeDash() {
  const stage = Object.fromEntries(rows("t_stage"));
  const names = Object.fromEntries(rows("t_names"));
  const parents = rows("t_parents");
  const leaves = rows("t_leaves");
  const placed = Object.values(stage).reduce((a, b) => a + b, 0);
  const sizes = leaves.map((r) => r[4]);
  const median = sizes.length ? [...sizes].sort((a, b) => a - b)[Math.floor(sizes.length / 2)] : null;
  const maxLeaf = Math.max(...sizes, 1);
  const leafRow = (r) => `<tr><td><span class="cellbar" style="--w:${(100 * r[4]) / maxLeaf}%"></span>${esc(r[3] || "unnamed")}</td><td class="mono tech">${r[0]}.${r[1]}.${r[2]}</td><td class="r">${n(r[4])}</td></tr>`;
  return `<div class="tiles">
      ${tile({ k: "Papers placed", v: n(placed), s: `${n(stage[1] || 0)} in the tree, ${n(stage[0] || 0)} projected only`, icon: "tree", dom: "dom-tree", hot: true })}
      ${tile({ k: "Grandparents", v: n(names[0]), s: "the top level", icon: "tree", dom: "dom-tree" })}
      ${tile({ k: "Parents", v: n(names[1]), s: "sixteen per grandparent", icon: "tree", dom: "dom-tree" })}
      ${tile({ k: "Leaves", v: n(names[2]), s: "sixteen per parent", icon: "tree", dom: "dom-tree" })}
      ${tile({ k: "Median leaf", v: n(median), s: "papers per leaf", icon: "layers", dom: "dom-over" })}
    </div><div class="charts">
      ${chart("The topic tree", `<div class="tt-bar"><input type="search" id="ttq" placeholder="Find a topic at any level&hellip;" value="${esc(tree.query)}">
        <span class="tech" id="ttn"></span><span class="spacer"></span>
        <div class="seg sm"><button data-ts="size" class="${tree.sort === "size" ? "on" : ""}">By size</button><button data-ts="name" class="${tree.sort === "name" ? "on" : ""}">By name</button></div>
        <button class="btn sm" id="ttall">Open every grandparent</button><button class="btn sm" id="ttnone">Collapse</button></div>
        <div class="tt-wrap"><div class="tt" id="tt"><svg class="tt-links" id="ttl"></svg></div></div>`,
        { w: "w12", q: "t_leaves", dom: "dom-tree", icon: "tree", sub: "Click a topic to open it. The bar beside each topic is its share of the topic above it, and hovering gives the counts." })}
      ${chart("Parent sizes", cols(logBins(parents.map((r) => r[3]), 12), { label: (k) => `about ${n(k)} papers` }), { q: "t_parents", dom: "dom-tree", icon: "layers", sub: "How many of the 256 parents hold how many papers, on a log scale." })}
      ${chart("Leaf sizes", cols(logBins(sizes, 16), { label: (k) => `about ${n(k)} papers` }), { q: "t_leaves", dom: "dom-papers", icon: "layers", sub: "The same for the 4,096 leaves." })}
      ${chart("Largest parents", hbars(parents.slice(0, 15).map((r) => [r[2] || `${r[0]}.${r[1]}`, r[3]]), { total: placed }), { q: "t_parents", dom: "dom-tree", icon: "tree" })}
      ${chart("Largest and smallest leaves", `<table class="dtable"><tr><th>leaf</th><th>id</th><th class="r">papers</th></tr>
        ${leaves.slice(0, 8).map(leafRow).join("")}<tr class="gap"><td colspan="3">…</td></tr>${leaves.slice(-5).map(leafRow).join("")}</table>`, { q: "t_leaves", dom: "dom-tree", icon: "layers" })}
    </div>`;
}

// -- the explorable topic tree ---------------------------------------------------------------

const TREE_ROW = 28;
// a leaf name runs to two lines, since the median one is 66 characters
const LEAF_ROW = 40;
// between one depth and the next, where the links run, and between chips sharing a row
const TREE_GAP = 32;
const TREE_COL = 14;
// how wide each depth's chip is, and how wide a chip sharing a row may stretch to fill it
const TREE_W = [150, 225, 250, 250];
const TREE_MAX = [150, 300, 340, 420];
// leaves hang below their parent rather than beside it, since three columns of ancestors
// leave too little width for names this long
const LEAF_INDENT = 30;
const depthX = (d) => TREE_W.slice(0, d).reduce((a, w) => a + w + TREE_GAP, 0);
const tree = { open: new Set(["root"]), query: "", sort: "size", hits: new Set(), shown: new Map(), links: new Map(), groups: new Map(), at: new Map(), width: 0 };

function treeData() {
  const stage = Object.fromEntries(rows("t_stage"));
  const root = { id: "root", name: "Every paper", n: Object.values(stage).reduce((a, b) => a + b, 0), depth: 0, children: [], code: "" };
  const byId = { root };
  rows("t_grand").forEach(([l0, name, n]) => {
    const node = { id: `g${l0}`, name: name || `topic ${l0}`, n, depth: 1, children: [], code: `${l0}`, up: root };
    root.children.push(node);
    byId[node.id] = node;
  });
  for (const [l0, l1, name, n] of rows("t_parents")) {
    const up = byId[`g${l0}`];
    if (!up) continue;
    const node = { id: `p${l0}.${l1}`, name: name || `topic ${l0}.${l1}`, n, depth: 2, children: [], code: `${l0}.${l1}`, up };
    up.children.push(node);
    byId[node.id] = node;
  }
  for (const [l0, l1, l2, name, n] of rows("t_leaves")) {
    const up = byId[`p${l0}.${l1}`];
    if (!up) continue;
    const node = { id: `l${l0}.${l1}.${l2}`, name: name || `topic ${l0}.${l1}.${l2}`, n, depth: 3, children: [], code: `${l0}.${l1}.${l2}`, up };
    up.children.push(node);
    byId[node.id] = node;
  }
  // each grandparent keeps one hue for its whole branch, so an open branch reads as one thing
  const ordered = [...root.children].sort((a, b) => b.n - a.n);
  ordered.forEach((g, i) => { g.t = Math.round((100 * i) / Math.max(1, ordered.length - 1)); });
  for (const node of Object.values(byId)) {
    let g = node;
    while (g.depth > 1) g = g.up;
    node.t = node.depth === 0 ? 0 : g.t;
  }
  return { root, byId };
}

function sorted(children) {
  return [...children].sort((a, b) => (tree.sort === "name" ? a.name.localeCompare(b.name) : b.n - a.n));
}

// an open topic takes a band of its own, and the closed siblings between open ones share rows,
// as many to a row as the width holds, so the tree grows sideways before it grows down
function treeLayout(root, width) {
  const out = [];
  const groups = [];
  let y = 0;
  const pack = (run, d, x0, below) => {
    if (!run.length) return;
    // the panel behind a run reaches 6px past its last chip
    const room = Math.max(TREE_W[d], width - x0 - 6);
    const per = Math.max(1, Math.floor((room + TREE_COL) / (TREE_W[d] + TREE_COL)));
    const w = Math.min(TREE_MAX[d], Math.floor((room - (per - 1) * TREE_COL) / per));
    const h = d === TREE_W.length - 1 ? LEAF_ROW : TREE_ROW;
    if (run.length > 1) {
      const across = Math.min(run.length, per);
      groups.push({ id: run[0].id, x: x0 - 6, y, w: across * (w + TREE_COL) - TREE_COL + 12, h: Math.ceil(run.length / per) * h, t: run[0].t });
    }
    run.forEach((kid, i) => {
      const c = i % per;
      if (i && !c) y += h;
      Object.assign(kid, { x: x0 + c * (w + TREE_COL), y: y + h / 2, w, h, kids: [], head: !c, below });
      out.push(kid);
    });
    y += h;
  };
  const walk = (node) => {
    const kids = tree.open.has(node.id) ? sorted(node.children) : [];
    Object.assign(node, { x: depthX(node.depth), w: TREE_W[node.depth], h: TREE_ROW, kids, head: true, below: false });
    out.push(node);
    const d = node.depth + 1;
    if (!kids.length || d === TREE_W.length - 1) {
      node.y = y + TREE_ROW / 2;
      y += TREE_ROW;
      pack(kids, d, node.x + LEAF_INDENT, true);
      return;
    }
    let run = [];
    for (const kid of kids) {
      if (!tree.open.has(kid.id) || !kid.children.length) {
        run.push(kid);
        continue;
      }
      pack(run, d, depthX(d), false);
      run = [];
      walk(kid);
    }
    pack(run, d, depthX(d), false);
    node.y = (kids[0].y + kids[kids.length - 1].y) / 2;
  };
  walk(root);
  return { nodes: out, groups, height: y };
}

function mountTree() {
  const host = $("tt");
  if (!host) return;
  const { root, byId } = treeData();
  tree.byId = byId;
  tree.shown = new Map();
  tree.links = new Map();
  tree.groups = new Map();
  $("ttq").oninput = (e) => { tree.query = e.target.value; findTopics(); drawTree(); };
  for (const b of document.querySelectorAll("[data-ts]")) {
    b.onclick = () => {
      tree.sort = b.dataset.ts;
      for (const x of document.querySelectorAll("[data-ts]")) x.classList.toggle("on", x === b);
      drawTree();
    };
  }
  if (tree.watch) tree.watch.disconnect();
  tree.width = 0;
  tree.watch = new ResizeObserver(() => { if (Math.abs(host.clientWidth - tree.width) > 1) drawTree(); });
  tree.watch.observe(host);
  $("ttall").onclick = () => { tree.open = new Set(["root", ...root.children.map((g) => g.id)]); drawTree(); };
  $("ttnone").onclick = () => { tree.open = new Set(["root"]); tree.query = ""; $("ttq").value = ""; tree.hits = new Set(); drawTree(); };
  tree.root = root;
  if (tree.query) findTopics();
  drawTree();
}

// a search opens the path to every match, up to a cap that keeps the tree readable
function findTopics() {
  const q = tree.query.trim().toLowerCase();
  tree.hits = new Set();
  if (!q) { $("ttn").textContent = ""; return; }
  const found = Object.values(tree.byId).filter((node) => node.depth > 0 && node.name.toLowerCase().includes(q));
  for (const node of found.slice(0, 40)) {
    tree.hits.add(node.id);
    for (let up = node.up; up; up = up.up) tree.open.add(up.id);
  }
  $("ttn").textContent = found.length ? `${n(found.length)} match${found.length === 1 ? "" : "es"}${found.length > 40 ? ", the first 40 opened" : ""}` : "no topic matches";
}

// a link leaves the right edge of a topic for the one beside it, or drops from its dot to a
// leaf row below it, and both shapes keep one command list so a link can animate from its start
function linkPath(a, b) {
  if (b.below) {
    const x = a.x + 10;
    const top = a.y + 10;
    const r = Math.min(8, Math.max(0, b.y - top));
    return `M${x},${top} L${x},${b.y - r} Q${x},${b.y} ${x + r},${b.y} L${b.x},${b.y}`;
  }
  const x1 = a.x + a.w;
  const mid = (x1 + b.x) / 2;
  return `M${x1},${a.y} C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`;
}

function drawTree() {
  const host = $("tt");
  const svg = $("ttl");
  if (!host) return;
  tree.width = host.clientWidth;
  const { nodes, groups, height } = treeLayout(tree.root, tree.width);
  host.style.height = `${height + 8}px`;
  svg.setAttribute("height", String(height + 8));

  // closed siblings sharing rows sit on one faint panel, so a row reads as one group
  const panels = new Set(groups.map((g) => g.id));
  for (const g of groups) {
    let el = tree.groups.get(g.id);
    if (!el) {
      el = document.createElement("div");
      el.className = "tg";
      el.style.opacity = "0";
      el.style.transform = `translate(${g.x}px, ${g.y}px)`;
      host.prepend(el);
      tree.groups.set(g.id, el);
    }
    el.style.setProperty("--t", `${g.t}%`);
    requestAnimationFrame(() => {
      el.style.transform = `translate(${g.x}px, ${g.y}px)`;
      el.style.width = `${g.w}px`;
      el.style.height = `${g.h}px`;
      el.style.opacity = "1";
    });
  }
  for (const [id, el] of tree.groups) {
    if (panels.has(id)) continue;
    el.style.opacity = "0";
    tree.groups.delete(id);
    setTimeout(() => el.remove(), 320);
  }
  const keep = new Set(nodes.map((node) => node.id));
  const origin = (node) => tree.at.get(node.up ? node.up.id : "root") || { x: 0, y: node.y };

  for (const node of nodes) {
    let el = tree.shown.get(node.id);
    const share = node.up ? node.n / node.up.n : 1;
    const internal = node.children.length > 0;
    const open = tree.open.has(node.id);
    if (!el) {
      el = document.createElement("button");
      el.className = `tn d${node.depth}`;
      const from = origin(node);
      el.style.transform = `translate(${from.x}px, ${from.y - node.h / 2}px)`;
      el.style.opacity = "0";
      el.onclick = () => {
        if (!internal) return;
        if (tree.open.has(node.id)) {
          const shut = (x) => { tree.open.delete(x.id); x.children.forEach(shut); };
          shut(node);
        } else tree.open.add(node.id);
        drawTree();
      };
      host.append(el);
      tree.shown.set(node.id, el);
    }
    el.style.setProperty("--t", `${node.t}%`);
    el.style.width = `${node.w}px`;
    el.style.height = `${node.h}px`;
    el.classList.toggle("tall", node.h === LEAF_ROW);
    el.classList.toggle("open", open && internal);
    el.classList.toggle("leaf", !internal);
    el.classList.toggle("hit", tree.hits.has(node.id));
    el.dataset.tip = `<b>${esc(node.name)}</b><br>${n(node.n)} papers${node.up ? ` · ${pct(node.n, node.up.n)} of ${esc(node.up.name)}` : ""}
      <br>${pct(node.n, tree.root.n)} of the corpus${node.code ? ` · topic ${node.code}` : ""}${internal ? `<br>${node.children.length} topics below, click to ${open ? "close" : "open"}` : ""}`;
    el.innerHTML = `<span class="tn-dot">${internal ? (open ? "&minus;" : "+") : ""}</span>
      <span class="tn-name">${esc(node.name)}</span><span class="tn-n">${short(node.n)}</span>
      ${node.up ? `<span class="tn-bar"><i style="width:${Math.max(2, share * 100)}%"></i></span>` : ""}`;
    requestAnimationFrame(() => {
      el.style.transform = `translate(${node.x}px, ${node.y - node.h / 2}px)`;
      el.style.opacity = "1";
    });
  }
  for (const [id, el] of tree.shown) {
    if (keep.has(id)) continue;
    const node = tree.byId[id];
    let up = node.up;
    while (up && !keep.has(up.id)) up = up.up;
    if (up) el.style.transform = `translate(${up.x}px, ${up.y - node.h / 2}px)`;
    el.style.opacity = "0";
    el.style.pointerEvents = "none";
    tree.shown.delete(id);
    setTimeout(() => el.remove(), 320);
  }

  const ns = "http://www.w3.org/2000/svg";
  const want = new Set();
  for (const node of nodes) {
    for (const kid of node.kids) {
      if (!kid.head) continue;
      want.add(kid.id);
      let path = tree.links.get(kid.id);
      if (!path) {
        path = document.createElementNS(ns, "path");
        const start = kid.below ? { x: node.x + 10, y: node.y + 10, below: true } : { x: node.x + node.w, y: node.y };
        path.style.d = `path("${linkPath(node, start)}")`;
        svg.append(path);
        tree.links.set(kid.id, path);
      }
      path.setAttribute("class", `tl${tree.hits.has(kid.id) ? " hit" : ""}`);
      path.style.setProperty("--t", `${kid.t}%`);
      const d = `path("${linkPath(node, kid)}")`;
      requestAnimationFrame(() => { path.style.d = d; });
    }
  }
  for (const [id, path] of tree.links) {
    if (want.has(id)) continue;
    path.style.opacity = "0";
    tree.links.delete(id);
    setTimeout(() => path.remove(), 320);
  }
  tree.at = new Map(nodes.map((node) => [node.id, { x: node.x, y: node.y }]));
}

function citeDash() {
  const ct = Object.fromEntries(rows("c_totals"));
  const { by, stage } = intentTotals();
  const scores = {};
  for (const [i, b, v] of rows("c_score")) (scores[i] = scores[i] || []).push([b, v]);
  const out = rows("c_out");
  const papersWith = out.reduce((a, r) => a + r[1], 0);
  const edges = (b) => Math.round(Math.exp(((b - 1) * Math.log(1000)) / 15));
  const top = rows("c_top");
  const maxTop = Math.max(...top.map((r) => r[3]), 1);
  return `<div class="tiles">
      ${tile({ k: "Citation rows", v: n(ct.rows), s: "one per occurrence in a paper", icon: "cite", dom: "dom-cite", hot: true })}
      ${tile({ k: "Resolved", v: n(ct.resolved), s: "point at a paper here", icon: "check", dom: "dom-cite", ring: ct.rows ? ct.resolved / ct.rows : 0, color: "--secondary" })}
      ${tile({ k: "Labeled", v: n(stage[1]), s: `${n(stage[0])} still waiting`, icon: "check", dom: "dom-cite", ring: ct.resolved ? stage[1] / ct.resolved : 0, color: "--green" })}
      ${tile({ k: "Papers citing", v: n(papersWith), s: "with at least one row", icon: "papers", dom: "dom-papers" })}
    </div><div class="charts">
      ${chart("Intent", donut(INTENTS.filter((i) => by[i]).map((i) => [i, by[i], INTENT_COLOR[i]]), { v: pct((by.extends || 0) + (by.critiques || 0) + (by.compares || 0), ct.resolved), k: "take a position" }), { q: "c_intent", dom: "dom-cite", icon: "cite", sub: "Resolved citations only. The center is the share that extends, critiques or compares." })}
      ${chart("How a citation was resolved", hbars(rows("c_resolved_by"), { total: ct.rows }), { q: "c_resolved_by", dom: "dom-cite", icon: "check" })}
      ${chart("Citations per citing paper", cols(out, { label: (b) => (b >= 30 ? "300 or more" : `${b * 10} to ${b * 10 + 9}`), ticks: ["0", "150", "300+"] }), { q: "c_out", dom: "dom-papers", icon: "papers" })}
      ${chart("Times each paper is cited", cols(rows("c_in"), { label: (b) => (b >= 16 ? "1,000 or more" : `${n(edges(b))} to ${n(edges(b + 1))}`), ticks: ["1", "30", "1,000+"] }), { q: "c_in", dom: "dom-cite", icon: "cite", sub: "Resolved citations, on a log scale." })}
      ${chart("Section the citation sits in, top 20", hbars(rows("c_section"), { total: ct.rows }), { q: "c_section", dom: "dom-over", icon: "layers" })}
      ${chart("Most cited papers", `<table class="dtable top"><tr><th></th><th>paper</th><th>year</th><th class="r">cited</th></tr>
        ${top.map(([id, t, d, v], i) => `<tr><td class="rk">${i + 1}</td><td><span class="cellbar" style="--w:${(100 * v) / maxTop}%"></span><a href="https://arxiv.org/abs/${esc(id)}" target="_blank">${esc(t)}</a></td><td class="tech">${esc(String(d).slice(0, 4))}</td><td class="r">${n(v)}</td></tr>`).join("")}</table>`, { q: "c_top", dom: "dom-cite", icon: "spark" })}
      ${chart("Labeler confidence by intent", `<div class="multiples">${INTENTS.filter((i) => scores[i]).map((i) =>
        `<div class="mult" style="--c1:color-mix(in oklab, ${hue(INTENT_COLOR[i])} 45%, var(--mantle));--c2:${hue(INTENT_COLOR[i])}"><div class="mult-h"><i style="background:${hue(INTENT_COLOR[i])}"></i>${i}<span>${n(by[i])}</span></div>${cols(scores[i], { label: (b) => `score ${((b - 1) / 20).toFixed(2)} to ${(b / 20).toFixed(2)}`, ticks: ["0", "0.5", "1"] })}</div>`).join("")}</div>`,
        { w: "w12", q: "c_score", dom: "dom-cite", icon: "check", sub: "How sure the labeler was of the intent it chose, one panel per intent." })}
    </div>`;
}

function storageDash() {
  const tables = rows("s_tables");
  const idx = rows("s_indexes");
  const db = (rows("s_database")[0] || [])[0];
  const disks = Object.entries((S.health || {}).disk || {});
  const maxIdx = Math.max(...idx.map((r) => r[2]), 1);
  // these four stay distinct in every palette, which secondary and yellow do not
  const tableColor = ["--primary", "--sky", "--green", "--yellow"];
  return `<div class="tiles">${tile({ k: "Database", v: bytes(db), s: "on disk, indexes included", icon: "disk", dom: "dom-store", hot: true })}
      ${disks.map(([k, v]) => tile({ k: esc(k), v: v.free != null ? bytes(v.free) : "–", s: v.total ? `free of ${bytes(v.total)}` : esc(v.note), icon: "disk", dom: "dom-store", ring: v.total ? 1 - v.free / v.total : null, color: v.ok ? "--yellow" : "--red" })).join("")}</div>
    <div class="charts">
      ${chart("Space by table", donut(tables.map(([t, total], i) => [t, total, tableColor[i % tableColor.length]]), { v: bytes(db), k: "in total" }, { fmt: bytes }), { q: "s_tables", dom: "dom-store", icon: "disk" })}
      ${chart("Tables", `<table class="dtable"><tr><th>table</th><th class="r">rows</th><th class="r">data</th><th class="r">indexes</th><th class="r">total</th></tr>
        ${tables.map(([t, total, heap, ix, r], i) => `<tr><td><i class="sw" style="background:${hue(tableColor[i % tableColor.length])}"></i>${esc(t)}</td><td class="r">${n(r)}</td><td class="r">${bytes(heap)}</td><td class="r">${bytes(ix)}</td><td class="r"><b>${bytes(total)}</b></td></tr>`).join("")}</table>`, { q: "s_tables", dom: "dom-store", icon: "layers" })}
      ${chart("Indexes", `<table class="dtable"><tr><th>index</th><th>table</th><th class="r">size</th><th class="r">scans</th></tr>
        ${idx.map(([i, t, size, scans]) => `<tr><td class="mono" style="font-size:11.5px"><span class="cellbar" style="--w:${(100 * size) / maxIdx}%"></span>${esc(i)}</td><td class="tech">${esc(t)}</td><td class="r">${bytes(size)}</td><td class="r">${n(scans)}</td></tr>`).join("")}</table>`, { w: "w12", q: "s_indexes", dom: "dom-store", icon: "disk" })}
    </div>`;
}

// -- the download manager --------------------------------------------------------------------

function stageStatus(st) {
  const live = running();
  const on = live && (live.steps[live.current] || {}).stage === st.key;
  if (on) return { led: "run", text: live.status === "stopping" ? "stopping" : "running", job: live };
  const queued = (S.jobs || []).find((j) => ["queued", "scheduled", "waiting"].includes(j.status) && j.steps.some((s) => s.stage === st.key));
  if (queued) return { led: "warn", text: queued.status === "waiting" ? `waiting: ${queued.reason}` : "queued", job: queued };
  const last = st.last;
  if (!last) return { led: "idle", text: "never run from here" };
  const bad = ["failed", "blocked", "interrupted", "cancelled"].includes(last.status);
  return { led: bad ? "bad" : last.status === "done" ? "ok" : "idle", text: `last run ${last.status}` };
}

function download(full) {
  const host = $("p-download");
  const tables = S.tables || {};
  const mine = (S.stages || []).filter((s) => s.table === where.table);
  if (!mine.some((s) => s.key === where.stage)) where.stage = (mine[0] || {}).key;
  const st = mine.find((s) => s.key === where.stage);
  if (full || !$("stagebox") || $("stagebox").dataset.key !== where.stage) {
    const delta = mine.filter((s) => !s.alone && (s.delta || s.table === "export"));
    host.innerHTML = `<div class="subbar"><div class="seg">${Object.entries(tables).map(([k, l]) => {
      const ss = (S.stages || []).filter((s) => s.table === k);
      const busy = ss.some((s) => stageStatus(s).led === "run");
      return `<button data-tb="${k}" class="${where.table === k ? "on" : ""}">${busy ? '<i class="led run"></i>' : ""}${l}</button>`;
    }).join("")}</div><span class="spacer"></span>
      <button class="btn" id="dryevery">Dry run every stage</button><button class="btn primary" id="runevery">Run every stage</button></div>
      <div class="tablebar"><b>${esc(tables[where.table] || "")}</b>
        <button class="btn sm" id="runtable">Run all ${delta.length} in order</button><button class="btn sm" id="drytable">Dry run all ${delta.length}</button>
        <span class="tech">A job runs its stages in order and stops at the first one that fails.</span></div>
      <div class="pipe" id="pipe"></div><div id="stagebox" data-key="${esc(where.stage)}"></div>`;
    for (const b of host.querySelectorAll("[data-tb]")) b.onclick = () => go({ table: b.dataset.tb, stage: `${b.dataset.tb}.0` });
    $("runevery").onclick = () => confirm("Run every stage in order, starting with the snapshot?") && act("/api/run", { all: true }, openRunning);
    $("dryevery").onclick = () => dry({ all: true }, where.stage);
    $("runtable").onclick = () => confirm(`Run all ${tables[where.table].toLowerCase()} stages in order?`) && act("/api/run", { table: where.table }, openRunning);
    $("drytable").onclick = () => dry({ table: where.table }, where.stage);
    if (st) stagePanel(st);
  }
  pipe(mine);
  if (st) stageLive(st);
}

function pipe(mine) {
  $("pipe").innerHTML = mine.map((s, i) => {
    const status = stageStatus(s);
    const live = status.led === "run" ? stepProgress(status.job, s.key) : null;
    const w = live && live.done != null && live.total ? (100 * live.done) / live.total : null;
    const facts = s.kind === "fetch"
      ? ((S.remote || {}).version ? `<b>v${esc(S.remote.version)}</b> on Kaggle` : "&nbsp;")
      : s.waiting != null ? `<b>${n(s.waiting)}</b> waiting` : "&nbsp;";
    return `${i ? '<span class="arrow">&rarr;</span>' : ""}<button class="node st-${status.led}${s.key === where.stage ? " on" : ""}" data-st="${s.key}" data-tip="${esc(status.text)}">
      <div class="node-k"><span class="step">${String(i + 1).padStart(2, "0")}</span>${esc(s.key)}<i class="led ${status.led}"></i></div><div class="node-t">${esc(s.label)}</div>
      <div class="node-w">${facts}</div>
      ${live ? `<div class="bar live${w == null ? " busy" : ""}"><i style="width:${w ?? 0}%"></i></div>` : ""}</button>`;
  }).join("");
  for (const b of $("pipe").querySelectorAll("[data-st]")) b.onclick = () => go({ stage: b.dataset.st });
}

function scriptFlags(st) {
  const found = (S.scripts || []).find((s) => s.path === st.script);
  return found ? found.flags.filter((f) => !st.args.includes(f.flag)) : [];
}

function stagePanel(st) {
  const flags = scriptFlags(st);
  $("stagebox").innerHTML = `<div class="stage">
    <div class="stage-h"><i class="led" id="sled"></i><h2>${esc(st.label)}</h2><span class="state" id="sstate"></span>
      <div class="btns"><button class="btn" id="sdry">Dry run</button><button class="btn primary" id="srun">${st.alone ? "Publish" : "Run"}</button>
        <button class="btn danger" id="sstop" style="display:none">Stop</button></div></div>
    <div class="cmd">${esc(st.command)}</div>
    <div class="facts" id="sfacts"></div>
    <div class="what"><span>Reads</span><span>${esc(st.reads)}</span><span>Writes</span><span>${esc(st.writes)}</span>
      ${st.fetches ? `<span>Fetches</span><span>${esc(st.fetches)}</span>` : ""}${st.note ? `<span>Note</span><span>${esc(st.note)}</span>` : ""}</div>
    <div class="prog" id="sprog"></div>
    ${flags.length ? `<details class="flagbox"><summary class="console-h" style="cursor:pointer">Extra flags for this run<span class="tech">${flags.length} from --help</span></summary>
      <div class="flags">${flags.map((f) => f.value
        ? `<label>${esc(f.flag)} <input type="text" data-flag="${esc(f.flag)}" placeholder="${esc(f.value)}"><span class="fh">${esc(f.help)}</span></label>`
        : `<label class="check"><input type="checkbox" data-flag="${esc(f.flag)}"> ${esc(f.flag)} <span class="fh">${esc(f.help)}</span></label>`).join("")}</div></details>` : ""}
    <div class="console-h"><i class="led" id="cled"></i><span id="ctitle">Console</span><span class="tech" id="cmeta"></span></div>
    <pre class="console" id="console"></pre></div>`;
  showing = { key: "", mode: "", job: "", offset: 0, text: "" };
  $("srun").onclick = () => {
    const args = [];
    for (const el of $("stagebox").querySelectorAll("[data-flag]")) {
      if (el.type === "checkbox" && el.checked) args.push(el.dataset.flag);
      if (el.type === "text" && el.value.trim()) args.push(el.dataset.flag, el.value.trim());
    }
    const ask = st.key === "citations.relabel" ? "Relabel every resolved citation? This resets them all and takes about 11 hours."
      : st.alone ? "Publish the newest release to Hugging Face? Every changed file is uploaded and each repo is tagged."
      : `Run ${st.label.toLowerCase()} now?`;
    if (confirm(ask)) { delete dryText[st.key]; act("/api/run", { stage: st.key, args }); }
  };
  // publishing has a real dry run, which asks the hub what would change
  $("sdry").onclick = () => st.alone ? act("/api/run", { stage: st.key, args: ["--dry-run"] }) : dry({ stage: st.key }, st.key);
}

function stageLive(st) {
  const status = stageStatus(st);
  $("sled").className = `led ${status.led}`;
  $("sstate").textContent = status.text;
  const on = status.led === "run";
  $("sstop").style.display = on ? "" : "none";
  $("sstop").onclick = () => confirm("Stop this job? The stage gets SIGINT and a minute to roll back.") && act(`/api/jobs/${status.job.id}/cancel`, {});
  const hub = S.hub || {};
  $("srun").disabled = on || (st.alone && !hub.token);
  const rate = (S.rates || {})[st.key];
  const last = st.last && (S.jobs || []).find((j) => j.id === st.last.job);
  const lastStep = last && last.steps[st.last.step];
  const repos = hub.repos || [];
  const shut = repos.filter((r) => !r.write).map((r) => r.id);
  $("sfacts").innerHTML = (st.alone ? [
    ["Signed in as", hub.user || "–", hub.note || (hub.role === "fineGrained" ? "a fine-grained token" : `a ${hub.role} token`)],
    ["Can publish to", hub.token ? `${repos.length - shut.length} of ${repos.length}` : "–", shut.length ? `not ${shut.join(", ")}` : repos.map((r) => r.id).join(", ")],
    ["Last run", lastStep ? lastStep.status : "–", lastStep ? `${when(lastStep.started)} · ${dur((lastStep.ended || S.now) - lastStep.started)}` : "never from this page"],
  ] : [
    st.kind === "fetch"
      ? ["Kaggle serves", (S.remote || {}).version ? `v${S.remote.version}` : "–", (S.remote || {}).published ? `published ${when(Date.parse(S.remote.published) / 1000)}` : "not asked yet"]
      : ["Waiting", st.waiting != null ? n(st.waiting) : "–", st.queue ? `queue: ${st.queue}` : "no queue"],
    ["Rate", rate ? pace(rate, st.unit) : "–", rate ? (st.unit === "bytes" ? "measured here" : "rows a second") : "not measured yet"],
    st.kind === "fetch"
      ? ["Size", (S.remote || {}).size ? bytes(S.remote.size) : "–", "unzipped, about 1.8 GB to download"]
      : ["Estimate", rate && st.waiting ? dur(st.waiting / rate) : "–", "for what is waiting"],
    ["Last run", lastStep ? lastStep.status : "–", lastStep ? `${when(lastStep.started)} · ${dur((lastStep.ended || S.now) - lastStep.started)}` : "never from this page"],
  ]).map(([k, v, s]) => `<div class="fact"><div class="fact-k">${k}</div><div class="fact-v">${esc(v)}</div><div class="fact-s">${esc(s)}</div></div>`).join("");
  $("sprog").innerHTML = on ? progressHtml(status.job, true)
    : `<div class="bar big"><i style="width:${lastStep && lastStep.status === "done" ? 100 : 0}%"></i></div>
       <div class="barline"><span>${lastStep ? `last run ${esc(lastStep.status)}${lastStep.note ? ": " + esc(lastStep.note) : ""}` : "not running"}</span><span>${lastStep && lastStep.code != null ? `exit ${lastStep.code}` : ""}</span></div>`;
  follow(st, status);
}

// -- the console -----------------------------------------------------------------------------

function paint(text) {
  return esc(text).split("\n").map((l) => {
    if (/^(===|==) /.test(l) || /^dry run/.test(l)) return `<span class="l-head">${l}</span>`;
    if (/Traceback|Error|refus|FATAL|failed/i.test(l)) return `<span class="l-err">${l}</span>`;
    if (/^\s+\$ /.test(l) || /^\s{3}\w+\s{2,}/.test(l)) return `<span class="l-dim">${l}</span>`;
    if (/done|inserted|wrote|landed|placed|named/i.test(l)) return `<span class="l-ok">${l}</span>`;
    return l;
  }).join("\n");
}

function sectionOf(text, label) {
  const mark = `=== ${label} ===`;
  const at = text.lastIndexOf(mark);
  if (at < 0) return text;
  const rest = text.slice(at);
  const next = rest.indexOf("\n=== ", mark.length);
  return next < 0 ? rest : rest.slice(0, next);
}

async function follow(st, status) {
  const box = $("console");
  if (!box) return;
  if (dryText[st.key] && status.led !== "run") {
    if (showing.mode !== "dry" || showing.text !== dryText[st.key]) {
      showing = { key: st.key, mode: "dry", job: "", offset: 0, text: dryText[st.key] };
      $("ctitle").textContent = "Dry run";
      $("cmeta").textContent = "printed, nothing was started";
      $("cled").className = "led idle";
      box.innerHTML = paint(dryText[st.key]);
    }
    return;
  }
  const job = status.led === "run" ? status.job : st.last && (S.jobs || []).find((j) => j.id === st.last.job);
  if (!job) {
    if (showing.mode !== "empty") {
      showing = { key: st.key, mode: "empty", job: "", offset: 0, text: "" };
      $("ctitle").textContent = "Console";
      $("cmeta").textContent = "";
      box.innerHTML = `<span class="l-dim">This stage has not run from the manager yet. Dry run prints what it would do.</span>`;
    }
    return;
  }
  const label = (job.steps.find((s) => s.stage === st.key) || {}).label || st.label;
  if (showing.job !== job.id || showing.mode !== "log") showing = { key: st.key, mode: "log", job: job.id, offset: 0, text: "" };
  else if (status.led !== "run" && showing.offset) return;
  const got = await api(`/api/jobs/${job.id}/log?from=${showing.offset}`).catch(() => null);
  if (!got || showing.job !== job.id) return;
  showing.text += got.text;
  showing.offset = got.offset;
  const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  box.innerHTML = paint(sectionOf(showing.text, label));
  if (stick || status.led === "run") box.scrollTop = box.scrollHeight;
  $("ctitle").textContent = `${job.title}`;
  $("cmeta").textContent = `${job.id} · ${job.status}`;
  $("cled").className = `led ${status.led === "run" ? "run" : job.status === "done" ? "ok" : ["failed", "blocked"].includes(job.status) ? "bad" : "idle"}`;
}

async function dry(body, key) {
  try {
    const got = await api("/api/plan", body);
    dryText[key] = got.text;
    if (body.all || body.table === "papers" || body.stage === "papers.0") {
      dryText[key] += "\nThe snapshot stage's own --dry-run also scans the file against the store. Run it from the rail, or with Compare below.\n";
    }
    if (where.tab !== "download" || where.stage !== key) go({ tab: "download", table: key.split(".")[0], stage: key });
    else stageLive((S.stages || []).find((s) => s.key === key));
  } catch (e) {
    alert(e.message);
  }
}

// -- history ---------------------------------------------------------------------------------

function historyTab() {
  const jobs = S.jobs || [];
  $("p-history").innerHTML = jobs.length ? `<table class="jobs"><tr><th></th><th>job</th><th>started</th><th>took</th><th>result</th><th></th></tr>
    ${jobs.map((j) => {
      const first = (j.steps.find((s) => s.started) || {}).started;
      const led = j.status === "done" || j.status === "nothing new" ? "ok" : ["failed", "blocked"].includes(j.status) ? "bad" : ["running", "stopping"].includes(j.status) ? "run" : ["queued", "waiting", "scheduled"].includes(j.status) ? "warn" : "idle";
      return `<tr><td><i class="led ${led}"></i></td><td><b>${esc(j.title)}</b><div class="steps">${j.steps.map((s) => `<i class="${s.status}" data-tip="${esc(s.label)}: ${esc(s.status)}"></i>`).join("")}</div></td>
        <td class="tech">${when(first || j.created)}</td><td class="num">${first ? dur((j.ended || S.now) - first) : "–"}</td>
        <td>${esc(j.status)}${j.reason ? `<div class="tech">${esc(j.reason)}</div>` : ""}</td>
        <td style="white-space:nowrap">${["failed", "blocked", "cancelled", "interrupted"].includes(j.status) ? `<button class="btn sm" data-resume="${j.id}">Resume</button>` : ""}
          ${["queued", "waiting", "scheduled", "running"].includes(j.status) ? `<button class="btn sm danger" data-cancel="${j.id}">Stop</button>` : ""}
          <button class="btn sm" data-open="${j.id}">Open</button></td></tr>`;
    }).join("")}</table>` : `<div class="pending">Nothing has run from the manager yet.</div>`;
  for (const b of document.querySelectorAll("#p-history [data-resume]")) b.onclick = () => act(`/api/jobs/${b.dataset.resume}/resume`, {});
  for (const b of document.querySelectorAll("#p-history [data-cancel]")) b.onclick = () => act(`/api/jobs/${b.dataset.cancel}/cancel`, {});
  for (const b of document.querySelectorAll("#p-history [data-open]")) b.onclick = () => openJob(b.dataset.open);
}

// -- config ----------------------------------------------------------------------------------

// each group borrows one palette hue, the way the dashboard's areas do
const CFG_GROUPS = [
  ["Models", "--secondary", ["endpoints"]],
  ["MCP and retrieval", "--primary", ["serve", "retrieve", "miner", "miner.weights"]],
  ["Corpus and storage", "--yellow", ["database", "source", "download", "dataset", "export"]],
  ["Preprocessing", "--green", ["survey", "umap", "topics", "citation_labeler"]],
  ["Benchmarks", "--sky", ["surveymatch", "benchmarks"]],
  ["Manager", "--overlay1", ["manager"]],
];
const BUILT_TIP = "nix also builds the units, caddy or the sandbox around this, so a change applies fully only after a rebuild";

let CFG = null;
let EDIT = {};
let CFG_PROBLEMS = [];
let CFG_OPEN = new Set();
let CFG_QUERY = "";
let CFG_ONLY = false;
const REVEALED = {};

try { CFG_OPEN = new Set(JSON.parse(localStorage.getItem("cfg-open") || "[]")); } catch (e) { CFG_OPEN = new Set(); }

function keepOpen() {
  try { localStorage.setItem("cfg-open", JSON.stringify([...CFG_OPEN])); } catch (e) { /* private window */ }
}

async function configCall(method, path, body) {
  const r = await fetch(path, { method, headers: { "content-type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  const got = await r.json().catch(() => ({}));
  return { ok: r.ok, got };
}

async function configTab() {
  const { ok, got } = await configCall("GET", "/api/config");
  if (!ok) { $("p-config").innerHTML = `<div class="pending">${esc(got.error || "the config did not load")}</div>`; return; }
  CFG = got;
  configDraw();
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const edited = (sec, key) => EDIT[sec] !== undefined && EDIT[sec][key] !== undefined;
const editOf = (sec, key) => (edited(sec, key) ? EDIT[sec][key] : undefined);

function shown(k, v) {
  if (k.kind === "list") return (v || []).join(", ");
  return v == null ? "" : String(v);
}

function parsed(k, text) {
  if (k.kind === "list") {
    const items = text.split(",").map((x) => x.trim()).filter((x) => x !== "");
    return k.item === "int" || k.item === "float" ? items.map(Number) : items;
  }
  if (k.kind === "int" || k.kind === "float") return Number(text);
  return text;
}

function put(sec, key, value) { EDIT[sec] = { ...(EDIT[sec] || {}), [key]: value }; }

function drop(sec, key) {
  if (!EDIT[sec]) return;
  delete EDIT[sec][key];
  if (!Object.keys(EDIT[sec]).length) delete EDIT[sec];
}

// a value equal to the default reverts the key, and one equal to the saved override is no edit
function decide(sec, k, v) {
  if (v === undefined || same(v, k.default)) return k.overridden ? put(sec, k.key, null) : drop(sec, k.key);
  if (k.overridden && same(v, k.value)) return drop(sec, k.key);
  return put(sec, k.key, v);
}

function rowState(sec, k) {
  const v = editOf(sec, k.key);
  if (v === null) return "revert";
  if (v !== undefined) return "edit";
  return k.overridden ? "over" : "default";
}

function inputFor(sec, k) {
  const id = `${sec}.${k.key}`;
  const v = editOf(sec, k.key);
  if (k.kind === "bool") {
    const on = v === null ? k.default : v !== undefined ? v : k.value;
    return `<label class="cfg-switch"><input type="checkbox" data-cfg="${esc(id)}" ${on ? "checked" : ""}><span>${on ? "on" : "off"}</span></label>`;
  }
  if (k.secret) {
    const note = v === null ? "cleared on save" : k.value ? "set, type to replace" : "not set";
    const text = typeof v === "string" ? v : REVEALED[id] || "";
    return `<input class="cfg-text" type="${REVEALED[id] ? "text" : "password"}" data-cfg="${esc(id)}" placeholder="${note}" value="${esc(text)}" autocomplete="off" spellcheck="false">
      ${k.value ? `<button class="cfg-icon" data-reveal="${esc(id)}" data-tip="Show the key">${REVEALED[id] ? "hide" : "show"}</button>` : ""}`;
  }
  const text = v === null ? "" : v !== undefined ? shown(k, v) : k.overridden ? shown(k, k.value) : "";
  const hint = shown(k, k.default);
  const type = k.kind === "int" || k.kind === "float" ? "number" : "text";
  return `<input class="cfg-text" type="${type}"${k.kind === "float" ? ' step="any"' : ""} data-cfg="${esc(id)}" placeholder="${esc(hint || "empty")}" value="${esc(text)}" spellcheck="false">`;
}

function rowHtml(sec, k) {
  const id = `${sec}.${k.key}`;
  const state = rowState(sec, k);
  const undo = state === "edit" ? "Discard this edit" : state === "over" ? "Revert to default on save" : state === "revert" ? "Keep the override" : "";
  return `<div class="cfg-row" data-row="${esc(id)}" data-state="${state}">
    <div class="cfg-k"><code>${esc(k.key)}</code>${k.secret ? '<span class="cfg-chip secret">secret</span>' : ""}${k.built ? `<span class="cfg-chip built" data-tip="${BUILT_TIP}">nix</span>` : ""}
      <div class="cfg-d" title="${esc(k.description)}">${esc(k.description)}</div></div>
    <div class="cfg-in">${inputFor(sec, k)}</div>
    <div class="cfg-act">${undo ? `<button class="cfg-icon" data-undo="${esc(id)}" data-tip="${undo}">&#8634;</button>` : ""}</div></div>`;
}

function counts(sec) {
  const keys = sec.keys;
  return {
    over: keys.filter((k) => k.overridden && editOf(sec.name, k.key) !== null).length,
    edits: Object.keys(EDIT[sec.name] || {}).length,
  };
}

function sectionHtml(sec, hue) {
  const c = counts(sec);
  const open = CFG_OPEN.has(sec.name);
  return `<section class="cfg-sec${open ? " open" : ""}" data-sec="${esc(sec.name)}" style="--hue: var(${hue})">
    <button class="cfg-sh" data-toggle="${esc(sec.name)}"><span class="cfg-chev">&#9656;</span><b>${esc(sec.description || sec.name)}</b><code>[${esc(sec.name)}]</code>
      <span class="cfg-n">${sec.keys.length} setting${sec.keys.length === 1 ? "" : "s"}</span>
      <span class="cfg-chip over" ${c.over ? "" : "hidden"}>${c.over} overridden</span><span class="cfg-chip edit" ${c.edits ? "" : "hidden"}>${c.edits} unsaved</span></button>
    <div class="cfg-rows">${sec.keys.map((k) => rowHtml(sec.name, k)).join("")}</div></section>`;
}

function grouped() {
  const byName = Object.fromEntries(CFG.sections.map((s) => [s.name, s]));
  const placed = new Set(CFG_GROUPS.flatMap((g) => g[2]));
  const groups = CFG_GROUPS.map(([title, hue, names]) => [title, hue, names.filter((n) => byName[n]).map((n) => byName[n])]);
  const rest = CFG.sections.filter((s) => !placed.has(s.name));
  if (rest.length) groups.push(["Other", "--overlay1", rest]);
  return groups.filter((g) => g[2].length);
}

function configDraw() {
  const c = CFG;
  const edits = Object.values(EDIT).reduce((a, keys) => a + Object.keys(keys).length, 0);
  const over = c.sections.reduce((a, s) => a + s.keys.filter((k) => k.overridden).length, 0);
  const focus = document.activeElement && document.activeElement.id === "cfgq";
  $("p-config").innerHTML = `<div class="cfg">
    <div class="cfg-bar">
      <input id="cfgq" class="cfg-search" type="search" placeholder="Search settings" value="${esc(CFG_QUERY)}" spellcheck="false">
      <label class="cfg-only"><input type="checkbox" id="cfgonly" ${CFG_ONLY ? "checked" : ""}> Changed only</label>
      <button class="btn sm" id="cfgopen">Expand all</button><button class="btn sm" id="cfgshut">Collapse all</button>
      <span class="cfg-status"><span class="cfg-chip over" ${over ? "" : "hidden"}>${over} overridden</span><span class="cfg-chip edit" id="cfgedits" ${edits ? "" : "hidden"}>${edits} unsaved</span></span>
      <button class="btn sm" id="cfgdrop" ${edits ? "" : "disabled"}>Discard</button>
      <button class="btn sm primary" id="cfgsave" ${edits ? "" : "disabled"}>Save</button>
    </div>
    <div class="cfg-warn" id="cfgwarn" ${edits ? "" : "hidden"}>Saving restarts graphiv-api and graphiv-frontend, which ends any MCP session in progress. A running stage keeps the values it started with.</div>
    ${CFG_PROBLEMS.length ? `<div class="cfg-alert"><b>The save was refused</b><ul>${CFG_PROBLEMS.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
      <button class="btn danger sm" id="cfgforce">Save without the checks</button></div>` : ""}
    ${c.stale.length ? `<div class="cfg-alert"><b>In the override but no longer a setting</b><ul>${c.stale.map((s) => `<li><code>${esc(s)}</code> <button class="btn sm" data-stale="${esc(s)}">Remove</button></li>`).join("")}</ul></div>` : ""}
    ${grouped().map(([title, hue, secs]) => `<div class="cfg-group" style="--hue: var(${hue})"><div class="rh"><i class="cfg-dot"></i>${esc(title)}</div>
      ${secs.map((s) => sectionHtml(s, hue)).join("")}</div>`).join("")}
    <div class="cfg-foot tech">Defaults from ${c.source === "nix" ? "the NixOS module" : "config.toml"}, overrides in <code>${esc(c.override)}</code>
      <button class="btn sm danger" id="cfgall" ${over ? "" : "disabled"}>Revert all</button></div></div>`;
  configWire();
  configFilter();
  if (focus) { const q = $("cfgq"); q.focus(); q.setSelectionRange(q.value.length, q.value.length); }
}

function configMarks(sec, key) {
  const s = CFG.sections.find((x) => x.name === sec);
  const k = s.keys.find((x) => x.key === key);
  const row = document.querySelector(`#p-config [data-row="${CSS.escape(`${sec}.${key}`)}"]`);
  row.dataset.state = rowState(sec, k);
  const st = row.dataset.state;
  const undo = st === "edit" ? "Discard this edit" : st === "over" ? "Revert to default on save" : st === "revert" ? "Keep the override" : "";
  row.querySelector(".cfg-act").innerHTML = undo ? `<button class="cfg-icon" data-undo="${esc(`${sec}.${key}`)}" data-tip="${undo}">&#8634;</button>` : "";
  const b = row.querySelector("[data-undo]");
  if (b) b.onclick = () => undoRow(b.dataset.undo);
  const c = counts(s);
  const head = document.querySelector(`#p-config [data-sec="${CSS.escape(sec)}"] .cfg-sh`);
  const [overChip, editChip] = head.querySelectorAll(".cfg-chip");
  overChip.hidden = !c.over; overChip.textContent = `${c.over} overridden`;
  editChip.hidden = !c.edits; editChip.textContent = `${c.edits} unsaved`;
  const edits = Object.values(EDIT).reduce((a, keys) => a + Object.keys(keys).length, 0);
  $("cfgedits").hidden = !edits; $("cfgedits").textContent = `${edits} unsaved`;
  $("cfgwarn").hidden = !edits;
  $("cfgsave").disabled = !edits; $("cfgdrop").disabled = !edits;
}

function specOf(id) {
  const cut = id.lastIndexOf(".");
  const sec = id.slice(0, cut);
  const key = id.slice(cut + 1);
  const s = CFG.sections.find((x) => x.name === sec);
  return { sec, key, k: s && s.keys.find((x) => x.key === key) };
}

function undoRow(id) {
  const { sec, key, k } = specOf(id);
  if (edited(sec, key)) drop(sec, key);
  else if (k.overridden) put(sec, key, null);
  delete REVEALED[id];
  configDraw();
}

function configFilter() {
  const q = CFG_QUERY.trim().toLowerCase();
  for (const group of document.querySelectorAll("#p-config .cfg-group")) {
    let groupShown = 0;
    for (const secEl of group.querySelectorAll(".cfg-sec")) {
      const s = CFG.sections.find((x) => x.name === secEl.dataset.sec);
      const title = `${s.description} ${s.name}`.toLowerCase();
      let rowsShown = 0;
      for (const row of secEl.querySelectorAll(".cfg-row")) {
        const k = s.keys.find((x) => `${s.name}.${x.key}` === row.dataset.row);
        const hit = !q || title.includes(q) || `${k.key} ${k.description}`.toLowerCase().includes(q);
        const changed = row.dataset.state !== "default";
        row.hidden = !(hit && (!CFG_ONLY || changed));
        rowsShown += row.hidden ? 0 : 1;
      }
      secEl.hidden = !rowsShown;
      secEl.classList.toggle("open", CFG_OPEN.has(s.name) || ((q || CFG_ONLY) && rowsShown > 0));
      groupShown += rowsShown ? 1 : 0;
    }
    group.hidden = !groupShown;
  }
}

function configWire() {
  const q = $("cfgq");
  q.oninput = () => { CFG_QUERY = q.value; configFilter(); };
  $("cfgonly").onchange = (e) => { CFG_ONLY = e.target.checked; configFilter(); };
  $("cfgopen").onclick = () => { CFG.sections.forEach((s) => CFG_OPEN.add(s.name)); keepOpen(); configFilter(); };
  $("cfgshut").onclick = () => { CFG_OPEN.clear(); keepOpen(); configFilter(); };
  for (const b of document.querySelectorAll("#p-config [data-toggle]")) {
    b.onclick = () => {
      const name = b.dataset.toggle;
      const sec = b.closest(".cfg-sec");
      if (sec.classList.contains("open")) { CFG_OPEN.delete(name); sec.classList.remove("open"); }
      else { CFG_OPEN.add(name); sec.classList.add("open"); }
      keepOpen();
    };
  }
  for (const el of document.querySelectorAll("#p-config [data-cfg]")) {
    const { sec, key, k } = specOf(el.dataset.cfg);
    if (k.kind === "bool") {
      el.onchange = () => { decide(sec, k, el.checked); el.nextElementSibling.textContent = el.checked ? "on" : "off"; configMarks(sec, key); };
    } else if (k.secret) {
      el.oninput = () => {
        const text = el.value;
        if (text === "" || text === REVEALED[el.dataset.cfg]) drop(sec, key);
        else put(sec, key, text);
        configMarks(sec, key);
      };
    } else {
      el.oninput = () => { decide(sec, k, el.value === "" ? undefined : parsed(k, el.value)); configMarks(sec, key); };
    }
  }
  for (const b of document.querySelectorAll("#p-config [data-undo]")) b.onclick = () => undoRow(b.dataset.undo);
  for (const b of document.querySelectorAll("#p-config [data-reveal]")) {
    b.onclick = async () => {
      const id = b.dataset.reveal;
      if (REVEALED[id]) { delete REVEALED[id]; configDraw(); return; }
      const { sec, key } = specOf(id);
      const { ok, got } = await configCall("GET", `/api/config/reveal?section=${encodeURIComponent(sec)}&key=${encodeURIComponent(key)}`);
      if (ok) { REVEALED[id] = got.value; configDraw(); }
    };
  }
  for (const b of document.querySelectorAll("#p-config [data-stale]")) {
    b.onclick = async () => {
      const id = b.dataset.stale;
      const cut = id.lastIndexOf(".");
      const { ok, got } = await configCall("POST", "/api/config/revert", { section: id.slice(0, cut), key: id.slice(cut + 1) });
      if (ok) CFG = got;
      configDraw();
    };
  }
  $("cfgall").onclick = async () => {
    if (!confirm("Revert every setting to its default? This empties the override and restarts graphiv-api and graphiv-frontend.")) return;
    const { ok, got } = await configCall("POST", "/api/config/revert", {});
    if (ok) { CFG = got; EDIT = {}; CFG_PROBLEMS = []; }
    configDraw();
  };
  $("cfgdrop").onclick = () => { EDIT = {}; CFG_PROBLEMS = []; configDraw(); };
  const save = async (force) => {
    const bad = Object.entries(EDIT).flatMap(([s, keys]) => Object.entries(keys)
      .filter(([, v]) => (typeof v === "number" && Number.isNaN(v)) || (Array.isArray(v) && v.some((x) => typeof x === "number" && Number.isNaN(x))))
      .map(([k]) => `${s}.${k} needs a number`));
    if (bad.length) { CFG_PROBLEMS = bad; configDraw(); return; }
    $("cfgsave").disabled = true;
    $("cfgsave").textContent = "Checking";
    const { ok, got } = await configCall("PUT", "/api/config", { changes: EDIT, force });
    if (ok) { CFG = got; EDIT = {}; CFG_PROBLEMS = []; for (const id of Object.keys(REVEALED)) delete REVEALED[id]; }
    else CFG_PROBLEMS = got.problems || [got.error || "the save failed"];
    configDraw();
  };
  $("cfgsave").onclick = () => save(false);
  if ($("cfgforce")) $("cfgforce").onclick = () => { if (confirm("Save without the checks passing?")) save(true); };
}

document.addEventListener("keydown", (e) => {
  if (e.key !== "/" || where.tab !== "config") return;
  const tag = (document.activeElement && document.activeElement.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  e.preventDefault();
  $("cfgq").focus();
});

// -- the loop --------------------------------------------------------------------------------

async function poll() {
  try {
    S = await api("/api/state");
  } catch (e) {
    $("counted").textContent = "the manager is not answering";
    return;
  }
  const taken = JSON.stringify(S.stats_taken || {});
  if (taken !== (poll.taken || "")) {
    ST = await api("/api/stats").catch(() => ST);
    poll.taken = taken;
  }
  leds();
  rail();
  const count = (S.jobs || []).filter((j) => ["queued", "waiting", "scheduled", "running", "stopping"].includes(j.status)).length;
  $("tabs").querySelector('[data-t="history"]').innerHTML = `History${count ? `<span class="count">${count} active</span>` : ""}`;
  draw(false);
}

function start() {
  theme();
  readHash();
  for (const b of $("tabs").querySelectorAll("button")) b.onclick = () => go({ tab: b.dataset.t });
  $("recount").onclick = () => act("/api/refresh", {});
  window.onhashchange = () => { readHash(); draw(true); };
  poll().then(() => draw(true));
  setInterval(poll, POLL);
}

start();
