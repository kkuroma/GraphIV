// The research library: every saved run, its report with every citation opened to its paragraph,
// the tables and lineages the server kept, and the process of every tool call and thought that made it
const PALETTES = ["haruhana", "natsumikan", "akiba", "fuyuyuki"];
const SEEDS = { haruhana: "#F07898", natsumikan: "#E08030", akiba: "#C84040", fuyuyuki: "#78B4E8" };
const TABS = ["report", "papers", "process", "timeline", "check"];
const MARKER = /\[([A-Z][A-Za-z]*\d+(?:\s*,\s*[A-Z][A-Za-z]*\d+)*)\]/g;
// the palettes repeat hues across their tokens, so tools take their own categorical set from style.css
const TOOL_COLORS = {
  search_papers: "--t-search",
  recall_paper: "--t-recall",
  read_paper: "--t-read",
  list_citations: "--t-citations",
  trace_lineages: "--t-lineages",
  save_report: "--t-save",
};
// a term's color in a search's why-ranked bars, the rest grouped as other
const TERM_COLORS = { cos: "--sky", bm25: "--secondary", anc: "--green", cocite: "--yellow", rec: "--primary", other: "--overlay1", hub: "--red" };
const INTENTS = { extends: "extends", critiques: "critiques", compares: "compares with", uses: "uses", motivates: "is motivated by" };

let runs = [];
let filter = "";
let run = null;
let cites = {};

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// a client may pass its effort along with the question, which is not part of what was asked
const asked = (q) => String(q ?? "").replace(/\s*Effort:\s*\w+\.?\s*$/i, "");
const deTeX = (s) => String(s ?? "").replace(/\$([^$]*)\$/g, "$1").replace(/[{}\\]/g, "").trim();

function authors(a) {
  const parts = deTeX(a).split(/,\s*/).filter(Boolean);
  return parts.length > 3 ? parts.slice(0, 3).join(", ") + " et al." : parts.join(", ");
}

function duration(seconds) {
  const s = Math.round(Number(seconds) || 0);
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}

function when(iso) {
  const d = new Date(iso);
  return isNaN(d) ? iso || "" : d.toLocaleString("en-US", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

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
  // a link can fix the theme, which is how one url captures any of the eight
  const asked = new URLSearchParams(location.search);
  if (PALETTES.includes(asked.get("palette"))) palette = asked.get("palette");
  if (["light", "dark"].includes(asked.get("mode"))) mode = asked.get("mode");
  const holder = document.getElementById("swatches");
  for (const name of PALETTES) {
    const b = document.createElement("button");
    b.className = "swatch";
    b.dataset.palette = name;
    b.title = name;
    b.style.background = SEEDS[name];
    b.onclick = () => apply(name, document.documentElement.dataset.mode);
    holder.append(b);
  }
  document.getElementById("mode").onclick = () =>
    apply(document.documentElement.dataset.palette, document.documentElement.dataset.mode === "dark" ? "light" : "dark");
  apply(palette, mode);
}

// -- the library -----------------------------------------------------------------------------

function library() {
  const wanted = filter.trim().toLowerCase();
  const shown = runs.filter((r) => !wanted || `${r.question} ${r.tldr || ""}`.toLowerCase().includes(wanted));
  const cards = shown.map((r) => `
    <a class="card" href="#/run/${encodeURIComponent(r.slug)}/report">
      <h2>${esc(asked(r.question))}</h2>
      ${r.tldr ? `<p class="card-tldr">${esc(r.tldr.replace(MARKER, "").replace(/\s+([.,;])/g, "$1"))}</p>` : ""}
      <div class="meta">
        <span class="chip ${r.passed ? "ok" : "warn"}">${r.passed ? "checked" : unverified(r.unverified)}</span>
        <span class="chip">${duration(r.seconds)}</span>
        <span class="chip">${r.papers} papers read</span>
        ${r.tables ? `<span class="chip">${r.tables} table${r.tables > 1 ? "s" : ""}</span>` : ""}
        ${r.lineages ? `<span class="chip">${r.lineages} lineage${r.lineages > 1 ? "s" : ""}</span>` : ""}
      </div>
      <div class="byline">${esc(when(r.saved))}</div>
    </a>`).join("");
  document.getElementById("view").innerHTML = `
    <section class="hero">
      <h1>Research library</h1>
      <p class="sub">Every report GraphIV saved, each citation traced to the paragraph it came from.</p>
      <div class="searchwrap"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        <input id="q" type="search" placeholder="Search questions" value="${esc(filter)}"></div>
    </section>
    <div class="grid">${cards || '<p class="empty">No saved runs yet.</p>'}</div>`;
  const box = document.getElementById("q");
  box.oninput = () => { filter = box.value; library(); document.getElementById("q").focus(); };
}

// -- one run ---------------------------------------------------------------------------------

function paperOf(id) {
  return (run.papers || {})[id] || null;
}

// every quote label gets a number in reading order, the way a paper numbers its references
const unverified = (k) => `${k} line${k === 1 ? "" : "s"} unverified`;

// a run whose last save failed keeps its report, each line the check faulted grouped once
function faultGroups() {
  const c = run.check || {};
  if (c.passed) return [];
  const by = new Map();
  for (const f of c.faults || []) {
    const line = String(f.line || "").trim();
    if (!by.has(line)) by.set(line, []);
    by.get(line).push(f.where ? `${f.kind}: ${f.where}` : f.kind);
  }
  return [...by].map(([line, kinds]) => ({ line, kinds }));
}

// a token after each faulted line, inside a table row's last cell, which faultTags turns into a tag
function markFaults(md, groups) {
  groups.forEach((g, i) => {
    const at = g.line ? md.indexOf(g.line) : -1;
    if (at < 0) return;
    const end = at + (g.line.endsWith("|") ? g.line.lastIndexOf("|") : g.line.length);
    md = `${md.slice(0, end)}@@F${i}@@${md.slice(end)}`;
  });
  return md;
}

function faultTags(html, groups) {
  return html.replace(/ ?@@F(\d+)@@/g, (_, i) => `<span class="unver" title="${esc(groups[+i].kinds.join("\n"))}">unverified</span>`);
}

function numberMarkers(markdown) {
  cites = {};
  let next = 1;
  return markdown.replace(MARKER, (_, group) => group.split(",").map((raw) => {
    const label = raw.trim();
    if (!cites[label]) cites[label] = next++;
    return `<button class="cite" type="button" data-m="${label}">${cites[label]}</button>`;
  }).join(""));
}

function evidenceCard(label) {
  const ev = (run.evidence || {})[label];
  if (!ev) return `<p class="cp-s">No stored quote for ${esc(label)}.</p>`;
  const id = ev.arxiv_id;
  const p = paperOf(id);
  const head = p
    ? `<div class="ins-sec"><div class="cp-t">${esc(deTeX(p.title))}</div><div class="cp-m">${esc(authors(p.authors))} &middot; ${esc(p.date)} &middot; <a href="https://arxiv.org/abs/${id}" target="_blank">arXiv ${id}</a> &middot; <a href="pdf/${id}" target="_blank">pdf</a></div></div>`
    : `<div class="ins-sec"><div class="cp-t">arXiv ${esc(id)}</div></div>`;
  const where = ev.pid.startsWith("c-") ? `citing sentence about ${esc(ev.pid.slice(2))}` : `${esc(ev.section || "body")}, paragraph ${esc(ev.pid.slice(1))}`;
  if (ev.text.includes("\n") && ev.text.includes(" | ")) return head + `<div class="ins-sec"><div class="cp-s">${where}</div>${chunkTable(ev.text, ev.quote)}</div>`;
  let text = esc(proseTeX(ev.text));
  const quote = esc(proseTeX(ev.quote || ""));
  const at = quote ? text.toLowerCase().indexOf(quote.toLowerCase()) : -1;
  if (at >= 0) text = text.slice(0, at) + `<mark>${text.slice(at, at + quote.length)}</mark>` + text.slice(at + quote.length);
  return head + `<div class="ins-sec"><div class="cp-s">${where}</div><p class="cp-b">${text}</p></div>`;
}

const ARROWS = { uparrow: "\u2191", downarrow: "\u2193", to: "\u2192", pm: "\u00b1", in: "\u2208", times: "\u00d7", leq: "\u2264", geq: "\u2265", approx: "\u2248" };
const cellTeX = (s) => deTeX(String(s).replace(/\\h(?:dash)?line/g, "").replace(/\\(?:text|mathrm|mathbf|textbf|bf)\{([^}]*)\}/g, "$1").replace(/\\(uparrow|downarrow|to|pm|in|times|leq|geq|approx)\b/g, (_, w) => ARROWS[w])).replace(/[_^]/g, "");

// prose keeps its own underscores, so only the math spans in it are cleaned
const proseTeX = (s) => String(s ?? "").replace(/\$([^$]*)\$/g, (_, m) => cellTeX(m));

// a table chunk is its caption over one row a line, cells split by a bar, and a row the quote touches is marked
function chunkTable(text, quote) {
  const [caption, ...rows] = text.split("\n");
  const folded = (s) => cellTeX(s).toLowerCase().replace(/\s+/g, " ");
  const q = folded(quote || "");
  const body = rows.map((r) => {
    const cells = r.split(" | ");
    const hit = q && (q.includes(folded(r)) || folded(r).includes(q) || (cells.length > 1 && q.includes(folded(cells[0])) && q.includes(folded(cells[1]))));
    return `<tr class="${hit ? "hit" : ""}">${cells.map((c) => `<td>${esc(cellTeX(c))}</td>`).join("")}</tr>`;
  }).join("");
  return `<p class="cp-b">${esc(cellTeX(caption))}</p><div class="tablewrap"><table class="chunk">${body}</table></div>`;
}

// the inspector follows the citation under the pointer, and a click pins it until the next click
let pinned = null;

function inspect(label) {
  const box = document.getElementById("inspector");
  if (!box) return;
  const ev = (run.evidence || {})[label];
  const siblings = ev ? Object.entries(run.evidence || {}).filter(([l, e]) => e.arxiv_id === ev.arxiv_id && l !== label && e.claim) : [];
  const more = siblings.length
    ? `<div class="ins-sec ins-more"><div class="ins-h">Also from this paper</div>${siblings.slice(0, 8).map(([l, e]) => `<div class="ins-claim"><button class="cite" type="button" data-m="${esc(l)}">${esc(l)}</button> ${esc(e.claim)}</div>`).join("")}</div>`
    : "";
  box.innerHTML = `<div class="ins-label">${esc(label)}${pinned === label ? ' <span class="dim">pinned</span>' : ""}</div>${evidenceCard(label)}${ev && ev.claim ? `<div class="ins-sec"><div class="ins-h">The reader's claim</div><p class="ins-c">${esc(ev.claim)}</p></div>` : ""}${more}`;
  box.classList.add("on");
  wire(box);
}

function clearInspector() {
  pinned = null;
  const box = document.getElementById("inspector");
  if (box) {
    box.classList.remove("on");
    box.innerHTML = '<p class="dim">Point at a numbered citation to see the paragraph it rests on. Click to keep it here.</p>';
  }
}

function wire(root) {
  for (const a of root.querySelectorAll(".cite")) {
    a.onmouseenter = () => { if (!pinned) inspect(a.dataset.m); };
    a.onclick = () => { pinned = pinned === a.dataset.m ? null : a.dataset.m; inspect(a.dataset.m); };
  }
}

function cell(c) {
  const value = esc(c.value || "n/a");
  return c.label ? `${value} <button class="cite" type="button" data-m="${c.label}">&para;</button>` : `<span class="dim">${value}</span>`;
}

// a gap is a pair of options each used somewhere that no method combines
function untried(table) {
  const choice = table.schema.columns.filter((c) => c.kind === "choice");
  const out = [];
  for (let i = 0; i < choice.length; i++) {
    for (let j = i + 1; j < choice.length; j++) {
      const a = choice[i], b = choice[j];
      const used = (col) => new Set(table.rows.map((r) => ((r.cells[col.name] || {}).value || "").toLowerCase()).filter((v) => v && v !== "n/a"));
      const ua = used(a), ub = used(b);
      const seen = new Set(table.rows.map((r) => `${((r.cells[a.name] || {}).value || "").toLowerCase()}|${((r.cells[b.name] || {}).value || "").toLowerCase()}`));
      const rows = [...ua].map((x) => `<tr><th>${esc(x)}</th>${[...ub].map((y) => {
        const n = table.rows.filter((r) => ((r.cells[a.name] || {}).value || "").toLowerCase() === x && ((r.cells[b.name] || {}).value || "").toLowerCase() === y).length;
        return `<td class="${n ? "" : "gap"}">${n || "untried"}</td>`;
      }).join("")}</tr>`).join("");
      if (ua.size && ub.size) out.push(`<h4>${esc(a.name)} &times; ${esc(b.name)}</h4><div class="tablewrap"><table class="cross"><thead><tr><th></th>${[...ub].map((y) => `<th>${esc(y)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`);
      void seen;
    }
  }
  return out.join("");
}

function tablesHtml() {
  return (run.tables || []).map((t, n) => {
    const cols = t.schema.columns;
    const head = `<tr><th>method</th>${cols.map((c) => `<th title="${esc(c.description || "")}">${esc(c.name.replace(/_/g, " "))}</th>`).join("")}</tr>`;
    const body = t.rows.map((r) => {
      const p = paperOf(r.arxiv_id);
      const year = p ? String(p.date).slice(0, 4) : "";
      return `<tr><td><b>${esc(r.method)}</b><div class="dim">${esc(r.arxiv_id)}${year ? ", " + year : ""}</div></td>${cols.map((c) => `<td>${cell(r.cells[c.name] || {})}</td>`).join("")}</tr>`;
    }).join("");
    const gaps = untried(t);
    return `<h3>Table ${n + 1}</h3><div class="tablewrap"><table>${"<thead>" + head + "</thead><tbody>" + body + "</tbody>"}</table></div>
      ${gaps ? `<details class="gaps"><summary>Combinations no method in this table tries</summary><p class="dim">An empty cell is a lead, not a finding: the table holds only the papers this run read.</p>${gaps}</details>` : ""}`;
  }).join("");
}

function citedPapers() {
  const used = new Set();
  const text = `${run.tldr || ""}\n${run.report || ""}`;
  for (const m of text.matchAll(MARKER)) for (const l of m[1].split(",")) { const ev = (run.evidence || {})[l.trim()]; if (ev) used.add(ev.arxiv_id); }
  return used;
}

// papers by year, cited ones in the primary color and read-only ones recessive
function yearsChart() {
  const papers = Object.values(run.papers || {});
  if (papers.length < 2) return "";
  const cited = citedPapers();
  const years = {};
  for (const p of papers) { const y = String(p.date).slice(0, 4); (years[y] ||= { c: 0, r: 0 })[cited.has(p.arxiv_id) ? "c" : "r"]++; }
  const keys = Object.keys(years).sort();
  const top = Math.max(...keys.map((k) => years[k].c + years[k].r));
  const w = 26, gap = 10, h = 110, width = keys.length * (w + gap);
  const bars = keys.map((k, i) => {
    const x = i * (w + gap), c = years[k].c, r = years[k].r;
    const hc = (h * c) / top, hr = (h * r) / top;
    return `<g><title>${k}: ${c} cited, ${r} read only</title>
      <rect x="${x}" y="${h - hc - hr}" width="${w}" height="${Math.max(hr - (hr && hc ? 2 : 0), 0)}" rx="4" fill="var(--surface2)"/>
      <rect x="${x}" y="${h - hc}" width="${w}" height="${hc}" rx="4" fill="var(--primary)"/>
      <text x="${x + w / 2}" y="${h + 14}" text-anchor="middle">${k.slice(2)}</text></g>`;
  }).join("");
  return `<div class="ov-card"><div class="ov-h">Papers read, by year</div>
    <svg class="chart" viewBox="-4 -4 ${width + 8} ${h + 22}" preserveAspectRatio="xMinYMid meet">${bars}</svg>
    <div class="legend"><span><i style="background:var(--primary)"></i>cited</span><span><i style="background:var(--surface2)"></i>read only</span></div></div>`;
}

function topicsChart() {
  const counts = {};
  for (const p of Object.values(run.papers || {})) if (p.topic) counts[p.topic] = (counts[p.topic] || 0) + 1;
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 6);
  if (!rows.length) return "";
  const top = rows[0][1];
  return `<div class="ov-card"><div class="ov-h">Where they sit in the topic tree</div>${rows.map(([t, n]) => `<div class="hbar"><span class="hbar-t" title="${esc(t)}">${esc(t)}</span><span class="hbar-b"><span style="width:${(100 * n) / top}%"></span></span><span class="hbar-n">${n}</span></div>`).join("")}</div>`;
}

// every measured result a read reported, one row per paper, filterable by what it was measured on
let resultsOn = "";

function resultsTable() {
  const all = Object.entries(run.evidence || {}).filter(([, e]) => e.value);
  if (!all.length) return "";
  const sets = {};
  for (const [, e] of all) for (const d of (e.dataset || "").split(/,\s*(?:and\s+)?|\s+and\s+/)) if (d.trim()) sets[d.trim()] = (sets[d.trim()] || 0) + 1;
  const shared = Object.entries(sets).filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const shown = resultsOn ? all.filter(([, e]) => (e.dataset || "").includes(resultsOn)) : all;
  const byPaper = {};
  for (const [l, e] of shown) (byPaper[e.arxiv_id] ||= []).push([l, e]);
  const rows = Object.entries(byPaper).sort((a, b) => String((paperOf(a[0]) || {}).date).localeCompare(String((paperOf(b[0]) || {}).date))).map(([id, list]) => {
    const p = paperOf(id);
    const name = p ? deTeX(p.title) : id;
    const known = (d) => d && !/^(not specified|unspecified|n\/a|none|unknown)$/i.test(d.trim());
    const items = list.map(([l, e]) => `<div class="res"><span class="r-val">${esc(e.value)}</span> ${esc(e.metric)}${known(e.dataset) ? ` <span class="dim">on ${esc(e.dataset)}</span>` : ""} <button class="cite" type="button" data-m="${esc(l)}">${esc(l)}</button></div>`).join("");
    return `<tr><td class="r-paper"><span title="${esc(name)}">${esc(name)}</span>${p ? ` <span class="dim">${String(p.date).slice(0, 4)}</span>` : ""}</td><td>${items}</td></tr>`;
  }).join("");
  const chips = shared.length ? `<div class="r-filter">${["", ...shared.map(([d]) => d)].map((d) => `<button class="pill sm ${d === resultsOn ? "on" : ""}" data-set="${esc(d)}">${d ? esc(d) : "all"}</button>`).join("")}</div>` : "";
  return `<div class="ov-card wide" id="results"><div class="ov-h">Reported results <span class="dim">${all.length} from ${new Set(all.map(([, e]) => e.arxiv_id)).size} papers, each tied to its paragraph</span></div>${chips}
    <div class="tablewrap"><table class="results"><tbody>${rows}</tbody></table></div></div>`;
}

function wireResults() {
  for (const b of document.querySelectorAll(".r-filter button")) b.onclick = () => {
    resultsOn = b.dataset.set;
    document.getElementById("results").outerHTML = resultsTable();
    wireResults();
    wire(document.getElementById("results"));
  };
}

// the palettes repeat hues across their tokens, so intents take their own set from style.css
const INTENT_COLORS = { extends: "--i-extends", critiques: "--i-critiques", compares: "--i-compares", uses: "--i-uses", motivates: "--i-motivates", background: "--i-background" };

// one line as a vertical timeline, oldest first, each step marked with the intent its citer gave
function lineageGraph(line, label) {
  const steps = new Map(line.steps.map((s) => [`${s.citer}>${s.cited}`, s]));
  const rows = line.papers.map((p, i) => {
    const prev = i ? line.papers[i - 1].arxiv_id : null;
    const s = prev ? steps.get(`${p.arxiv_id}>${prev}`) || steps.get(`${prev}>${p.arxiv_id}`) : null;
    const link = i
      ? `<div class="tl-link ${s ? "" : "quiet"}" style="${s ? `--c:var(${INTENT_COLORS[s.intent] || "--overlay1"})` : ""}">${s ? `<span class="tl-intent">${esc(s.intent)}</span>${s.label ? ` <button class="cite" type="button" data-m="${esc(s.label)}">${esc(s.label)}</button>` : ""}` : ""}</div>`
      : "";
    const title = deTeX(p.title);
    return `${link}<div class="tl-node ${p.yours ? "" : "outside"}"><span class="tl-year">${esc(p.year)}</span><span class="tl-dot"></span><a class="tl-title" href="https://arxiv.org/abs/${esc(p.arxiv_id)}" target="_blank" title="${esc(title)}">${esc(title)}</a></div>`;
  }).join("");
  return `<div class="ln2"><div class="ov-h">Line ${esc(label)}</div><div class="vline">${rows}</div></div>`;
}

function lineagesCard() {
  const lines = (run.lineages || []).flatMap((call, c) => call.lines.map((line, i) => lineageGraph(line, `${c + 1}.${i + 1}`)));
  if (!lines.length) return "";
  const legend = Object.entries(INTENT_COLORS).map(([k, c]) => `<span><i style="background:var(${c})"></i>${k}</span>`).join("");
  return `<div class="ov-card wide"><div class="ov-h">How the field developed <span class="dim">filled dots are papers the run searched, hollow ones were reached through citations</span></div><div class="lines">${lines.join("")}</div><div class="legend">${legend}</div></div>`;
}

// each section of the report as its own card, opened, so a reader can fold what they have read
function sections(md) {
  const parts = md.split(/\n(?=## )/);
  return parts.map((part) => {
    const m = part.match(/^## (.*)\n?([\s\S]*)$/);
    const title = m ? m[1] : "";
    const html = marked.parse(m ? m[2] : part).replace(/<(p|li)>(Inference:)/g, '<$1 class="inference"><span class="inf-tag">Inference</span>');
    return title ? `<details class="sec" open><summary>${esc(title)}</summary><div class="sec-b">${html}</div></details>` : `<div class="sec-b">${html}</div>`;
  }).join("");
}

function reportTab() {
  // the page already carries the question as its title, and the TL;DR has its own card
  const raw = (run.report || "").replace(/^\s*>\s*\*\*TL;DR\*\*[^\n]*\n+/, "").replace(/^\s*#\s[^\n]*\n/, "");
  const groups = faultGroups();
  const md = numberMarkers(markFaults(`${run.tldr || ""}\n\n@@@\n\n${raw}`, groups));
  const [tldr, body] = md.split("\n\n@@@\n\n");
  const answer = run.tldr ? `<div class="answer"><div class="ov-h">The answer</div><p>${faultTags(marked.parseInline(tldr), groups)}</p></div>` : "";
  const tables = tablesHtml();
  const overview = [resultsTable(), yearsChart(), topicsChart(), lineagesCard(), tables ? `<div class="ov-card wide"><div class="ov-h">Methods compared</div>${tables}</div>` : ""].join("");
  const calls = run.calls || [];
  const count = (tool) => calls.filter((c) => c.tool === tool).length;
  const n = (k, word) => `${k} ${word}${k === 1 ? "" : "s"}`;
  const made = `<div class="made">Made from ${n(count("search_papers"), "search")}, ${n(count("recall_paper"), "lookup")} and ${n(count("read_paper"), "read")} of ${Object.keys(run.papers || {}).length} papers in ${duration(run.seconds)}, ${Object.keys(run.evidence || {}).length} quotes kept, ${(run.check || {}).passed ? "every citation checked" : `${unverified(groups.length)}, each marked where it sits`}. <a href="#/run/${encodeURIComponent(run.slug)}/process">See how</a></div>`;
  return `${answer}${made}<div class="overview">${overview}</div><article class="prose report">${faultTags(sections(body), groups)}</article>`;
}

// the run pivoted by paper: what each one was read for, and whether the report leaned on it
function papersTab() {
  const cited = citedPapers();
  const byPaper = {};
  for (const [label, ev] of Object.entries(run.evidence || {})) (byPaper[ev.arxiv_id] ||= []).push([label, ev]);
  const list = Object.values(run.papers || {}).sort((a, b) => (cited.has(b.arxiv_id) - cited.has(a.arxiv_id)) || String(a.date).localeCompare(String(b.date)));
  const rows = list.map((p) => {
    const evs = (byPaper[p.arxiv_id] || []).filter(([, ev]) => !ev.pid.startsWith("c-"));
    const claims = evs.map(([label, ev]) => `<li><button class="cite" type="button" data-m="${esc(label)}">${esc(label)}</button> ${esc(ev.claim || ev.quote)}${ev.value ? ` <span class="r-val">${esc(ev.value)}</span>` : ""}</li>`).join("");
    return `<div class="pcard ${cited.has(p.arxiv_id) ? "cited" : ""}">
      <div class="ref-t"><a href="https://arxiv.org/abs/${esc(p.arxiv_id)}" target="_blank">${esc(deTeX(p.title))}</a>${cited.has(p.arxiv_id) ? '<span class="cited-b">CITED</span>' : ""}</div>
      <div class="ref-m">${esc(authors(p.authors))} &middot; ${esc(p.date)} &middot; ${esc(p.arxiv_id)}${p.topic ? " &middot; " + esc(p.topic) : ""}</div>
      ${claims ? `<ul class="ref-q">${claims}</ul>` : '<div class="dim">read, nothing kept</div>'}
    </div>`;
  }).join("");
  return `<div><p class="dim">${list.length} papers read, ${cited.size} of them cited in the report, ${Object.keys(run.evidence || {}).length} labeled quotes.</p><div class="pgrid">${rows}</div></div>`;
}

// each term's share of a paper's log score, as one stacked bar per paper
function whyRanked(search) {
  if (!search) return "";
  const groups = (terms) => {
    const out = { cos: 0, bm25: 0, anc: 0, cocite: 0, rec: 0, other: 0, hub: 0 };
    for (const [t, v] of Object.entries(terms || {})) out[t in out ? t : "other"] += v;
    return out;
  };
  const rows = search.papers.slice(0, 12).map((p) => ({ p, g: groups(p.terms) }));
  const most = Math.max(0.0001, ...rows.map((r) => Object.entries(r.g).filter(([t]) => t !== "hub").reduce((a, [, v]) => a + v, 0)));
  const bars = rows.map(({ p, g }) => {
    const parts = Object.entries(g).filter(([t, v]) => t !== "hub" && v > 0)
      .map(([t, v]) => `<span title="${t} ${v.toFixed(2)}" style="width:${(100 * v) / most}%;background:var(${TERM_COLORS[t]})"></span>`).join("");
    return `<div class="why-row"><span class="why-t">${esc(deTeX(p.title))}</span><span class="why-bar">${parts}</span></div>`;
  }).join("");
  const legend = Object.entries(TERM_COLORS).filter(([t]) => t !== "hub").map(([t, c]) => `<span><i style="background:var(${c})"></i>${t}</span>`).join("");
  return `<div class="why"><div class="why-legend">${esc(search.mode)} recipe: ${legend}</div>${bars}</div>`;
}

// a title for an id from any reply that listed it, since only read papers are stored with metadata
function titles() {
  const out = {};
  for (const p of Object.values(run.papers || {})) out[p.arxiv_id] = deTeX(p.title);
  for (const c of run.calls || []) {
    for (const line of String(c.reply).split("\n")) {
      const m = line.match(/^\s*(\d{4}\.\d{4,5}|[a-z.-]+\/\d{7})\s*(?:\||\(\d{4}\))\s*(?:[\d-]+\s*\|\s*)?([^|]+)/);
      if (m && !out[m[1]]) out[m[1]] = deTeX(m[2].trim());
    }
  }
  return out;
}

const clip = (s, n) => (s.length > n ? s.slice(0, n).trimEnd() + "…" : s);
const listed = (reply) => String(reply).split("\n").filter((l) => /^(\d{4}\.\d{4,5}|[a-z.-]+\/\d{7}) \|/.test(l)).length;

// what one call did, in a sentence a reader without the tool docs can follow
function describe(c, title) {
  const a = c.args || {};
  const first = String(c.reply).split("\n")[0];
  const named = (id) => (title[id] ? `“${esc(clip(title[id], 90))}”` : esc(id));
  switch (c.tool) {
    case "search_papers": {
      const by = a.filter && a.filter.authors && a.filter.authors.length;
      const head = by ? `Listed papers by ${esc(a.filter.authors.join(", "))}`
        : a.arxiv_id ? `Searched for papers around ${named(a.arxiv_id)}, ${esc(a.mode)} mode`
        : `Searched “${esc(a.query)}”, ${esc(a.mode)} mode`;
      return [head, `${listed(c.reply)} papers returned`];
    }
    case "recall_paper":
      return [`Looked for the paper described as “${esc(a.description)}”`, listed(c.reply) ? `found ${named(first.split(" | ")[0])}` : "no paper matched"];
    case "read_paper": {
      const ids = a.arxiv_ids || [];
      return [`Read ${ids.length} paper${ids.length === 1 ? "" : "s"} for “${esc(clip(a.question || "", 140))}”`, ids.map(named).join(", ")];
    }
    case "list_citations":
      return [`Listed ${a.direction === "cited_by" ? "the papers citing" : "the references of"} ${named(a.arxiv_id)}`, esc(first)];
    case "trace_lineages":
      return [`Traced how ${(a.arxiv_ids || []).length} papers connect through their citations`, esc(first)];
    case "save_report":
      return [`Submitted the report for the check`, esc(first)];
    default:
      return [esc(c.tool), esc(first)];
  }
}

// what the model was doing outside any tool call, which includes composing the next call's arguments
const THINK = "--overlay1";
const WRITE = "--text";

const clock = (s) => { const t = Math.max(0, Math.round(s)); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`; };

function callNode(c, title, search) {
  const [head, sub] = describe(c, title);
  const params = Object.entries(c.args || {}).filter(([, v]) => v !== null && v !== "")
    .map(([k, v]) => `<tr><th>${esc(k)}</th><td>${k === "markdown" ? `<pre>${esc(v)}</pre>` : esc(typeof v === "string" ? v : JSON.stringify(v))}</td></tr>`).join("");
  const body = `<div class="st-h">Parameters</div><table class="st-params">${params}</table>
    ${search ? `<div class="st-h">Why each paper ranked</div>${whyRanked(search)}` : ""}
    <div class="st-h">Response</div><pre>${esc(c.reply)}</pre>`;
  return { start: c.start || 0, color: TOOL_COLORS[c.tool] || "--overlay1", kind: c.tool, label: `Tool: ${c.tool}`, head, sub: `${sub} &middot; ${duration(c.seconds || 0)}`, body };
}

function sayNode(e, start) {
  const thinking = e.kind === "reasoning";
  const text = String(e.text).trim();
  return { start, color: thinking ? THINK : WRITE, kind: thinking ? "thinking" : "writing", label: `Action: ${thinking ? "thinking" : "writing"}`, head: esc(clip(text.split("\n")[0], 180)), sub: "", body: `<div class="st-text">${esc(text)}</div>` };
}

// the run as columns in the order things happened, calls sent together sharing one column
function processColumns() {
  const calls = run.calls || [];
  const searches = (run.searches || []).slice();
  // a search that returned papers starts its reply with an arXiv id, and only those were kept
  const searchOf = (c) => (c.tool === "search_papers" && /^(\d{4}\.\d{4,5}|[a-z.-]+\/\d{7})/.test(String(c.reply)) ? searches.shift() : null);
  const title = titles();
  const cols = [];
  if (!run.session) {
    // without the client's transcript, calls whose times overlap are taken as sent together
    let end = -1;
    for (const c of calls) {
      const node = callNode(c, title, searchOf(c));
      if (cols.length && (c.start || 0) < end) cols[cols.length - 1].push(node);
      else cols.push([node]);
      end = Math.max(end, (c.start || 0) + (c.seconds || 0));
    }
    return cols;
  }
  const pending = calls.slice();
  const take = (tool) => { const i = pending.findIndex((c) => c.tool === tool); return i < 0 ? null : pending.splice(i, 1)[0]; };
  const started = calls.length ? calls[0].tool : "";
  // the client's clock starts before the server's, so the first paired call sets the offset
  let open = false, offset = 0, before = [], group = null;
  const say = (e) => { group = null; cols.push([sayNode(e, Math.max(0, e.at - offset))]); };
  for (const e of run.session.events) {
    if (e.kind !== "tool") {
      if (open) say(e); else before.push(e);
      continue;
    }
    // a session can hold earlier runs, so only the thinking right before this run's first call is its planning
    if (!open && e.tool !== started) { before = []; continue; }
    if (!open) { open = true; offset = e.at - (calls[0].start || 0); before.forEach(say); }
    const c = take(e.tool);
    if (!c) continue;
    if (!group) { group = []; cols.push(group); }
    group.push(callNode(c, title, searchOf(c)));
  }
  for (const c of pending) cols.push([callNode(c, title, searchOf(c))]);
  return cols;
}

let procCols = [];
let procAt = null;

function processTab() {
  const calls = run.calls || [];
  procCols = processColumns();
  procAt = null;
  const total = Math.max(1, run.seconds || 0, ...calls.map((c) => (c.start || 0) + (c.seconds || 0)));
  const spans = calls.map((c) => [c.start || 0, Math.min(total, (c.start || 0) + (c.seconds || 0))]);
  const parts = [["thinking", total - covered(spans), THINK], ...Object.keys(TOOL_COLORS).map((t) => [t, covered(spans.filter((_, k) => calls[k].tool === t)), TOOL_COLORS[t]])]
    .filter(([, v]) => v >= 0.5).sort((a, b) => b[1] - a[1]);
  const bar = parts.map(([name, v, c]) => `<span style="width:${(100 * v) / total}%;background:var(${c})" title="${name}, ${duration(v)}"></span>`).join("");
  const key = parts.map(([name, v, c]) => `<span><i style="background:var(${c})"></i>${name} <b>${duration(v)}</b></span>`).join("");
  const nodes = procCols.flat();
  const kinds = [...new Map(nodes.map((n) => [n.kind, n.color])).entries()];
  const legend = kinds.map(([k, c]) => `<span><i style="background:var(${c})"></i>${k}</span>`).join("");
  return `<div class="process"><h2>Thinking process</h2>
    <p class="dim">A breakdown of how this response was generated.</p>
    <div class="proc-top">
      <div class="proc-stats"><div><b>${calls.length}</b> tool calls</div><div><b>${run.session ? nodes.filter((n) => n.kind === "thinking").length : "n/a"}</b> thinking loops</div><div><b>${duration(total)}</b> spent</div></div>
      <div class="proc-time"><div class="st-h">Where the time went</div><div class="pt-bar">${bar}</div><div class="pt-key">${key}</div></div>
    </div>
    <div class="tree-legend">${legend}<span class="dim">Hover a dot for what it did, click it or use the arrow keys for everything.</span></div>
    <div class="tree" id="tree"></div>
    <div class="tree-panel" id="tree-panel"><p class="dim">Click a dot to see what was passed, what came back and what the model thought.</p></div></div>`;
}

const STEP = 32;
const LANE = 26;
const DOT = 8;

// the tree is drawn at the width it has, wrapping onto a new row like a line of text
function drawTree() {
  const host = document.getElementById("tree");
  if (!host) return;
  const width = host.clientWidth || 1000;
  const edge = DOT + 4;
  const per = Math.max(4, Math.floor((width - 2 * edge) / STEP) + 1);
  // the run spans the whole width, and one too long for it wraps at the narrowest spacing
  const gap = (width - 2 * edge) / Math.max(1, Math.min(per, procCols.length) - 1);
  const rows = [];
  for (let i = 0; i < procCols.length; i += per) rows.push(procCols.slice(i, i + per));
  let y = 0;
  const parts = [];
  const pos = [];
  rows.forEach((row, r) => {
    const tall = Math.max(...row.map((c) => c.length));
    const mid = y + edge + ((tall - 1) * LANE) / 2;
    // a single column, or the shorter last row of a wrapped run, sits in the middle
    const left = (width - (row.length - 1) * gap) / 2;
    row.forEach((col, k) => {
      const ci = r * per + k;
      const x = left + k * gap;
      pos[ci] = col.map((_, j) => [x, mid + (j - (col.length - 1) / 2) * LANE]);
      if (k) {
        const curve = ([ax, ay], [bx, by]) => `<path class="tr-edge" d="M${ax},${ay} C${(ax + bx) / 2},${ay} ${(ax + bx) / 2},${by} ${bx},${by}"/>`;
        const from = pos[ci - 1];
        const to = pos[ci];
        // one side single is a branch or a join, and two groups in a row meet at one point between them
        if (from.length === 1 || to.length === 1) for (const a of from) for (const b of to) parts.push(curve(a, b));
        else {
          const hub = [x - gap / 2, mid];
          for (const a of from) parts.push(curve(a, hub));
          for (const b of to) parts.push(curve(hub, b));
        }
      }
    });
    y += 2 * edge + (tall - 1) * LANE + 12;
  });
  const dots = procCols.map((col, ci) => col.map((n, j) => `<circle class="tr-dot" data-c="${ci}" data-j="${j}" cx="${pos[ci][j][0]}" cy="${pos[ci][j][1]}" r="${DOT}" style="fill:var(${n.color})"/>`).join("")).join("");
  host.innerHTML = `<svg width="${width}" height="${y}" role="img" aria-label="The run's steps in order">${parts.join("")}${dots}</svg>`;
  for (const d of host.querySelectorAll(".tr-dot")) {
    const at = [+d.dataset.c, +d.dataset.j];
    d.onmouseenter = (e) => { const n = procCols[at[0]][at[1]]; showTip(e, `<div class="tip-k" style="color:var(${n.color})">${esc(n.label)} &middot; ${clock(n.start)}</div><div>${n.head}</div>${n.sub ? `<div class="dim">${n.sub}</div>` : ""}`); };
    d.onmousemove = (e) => showTip(e);
    d.onmouseleave = hideTip;
    d.onclick = () => pick(at[0], at[1]);
  }
  if (procAt) mark();
}

function mark() {
  for (const d of document.querySelectorAll(".tr-dot")) d.classList.toggle("on", procAt && +d.dataset.c === procAt[0] && +d.dataset.j === procAt[1]);
}

function pick(c, j) {
  const n = procCols[c] && procCols[c][j];
  if (!n) return;
  procAt = [c, j];
  mark();
  const group = procCols[c].length > 1 ? ` &middot; ${j + 1} of ${procCols[c].length} sent together` : "";
  document.getElementById("tree-panel").innerHTML = `<div class="tp-head" style="--c:var(${n.color})"><span class="st-kind">${esc(n.label)}</span><span class="dim"> at ${clock(n.start)}, step ${c + 1} of ${procCols.length}${group}</span></div>
    <div class="st-head">${n.head}</div>${n.sub ? `<div class="st-sub">${n.sub}</div>` : ""}<div class="st-body">${n.body}</div>`;
}

window.addEventListener("resize", () => { drawTree(); drawGraph(); });

// arrows walk the tree once a dot is picked: across the run, and within calls sent together
document.addEventListener("keydown", (e) => {
  if (!procAt || !document.getElementById("tree") || /input|textarea/i.test(e.target.tagName)) return;
  const [c, j] = procAt;
  const moves = { ArrowRight: [c + 1, 0], ArrowLeft: [c - 1, 0], ArrowDown: [c, j + 1], ArrowUp: [c, j - 1] };
  const to = moves[e.key];
  if (!to || !procCols[to[0]] || !procCols[to[0]][to[1]]) return;
  e.preventDefault();
  pick(to[0], to[1]);
});

let tip = null;
function hideTip() { if (tip) tip.style.display = "none"; }

function showTip(e, html) {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "ptip";
    document.body.append(tip);
  }
  if (html !== undefined) tip.innerHTML = html;
  tip.style.display = "block";
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  let x = e.clientX + 14;
  let y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 14;
  if (y + h > innerHeight - 8) y = Math.max(8, e.clientY - h - 14);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

// spans merged so parallel calls count once
function merged(spans) {
  const out = [];
  for (const [a, b] of spans.slice().sort((x, y) => x[0] - y[0])) {
    if (out.length && a <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], b);
    else out.push([a, b]);
  }
  return out;
}
const covered = (spans) => merged(spans).reduce((t, [a, b]) => t + b - a, 0);

// -- the citation graph among the run's papers --

const ROLES = { cited: "--primary", read: "--t-read", found: "--overlay1" };
const ROLE_WORDS = { cited: "cited in the report", read: "read", found: "returned, not read" };
// the two most common intents bury the rest, so they start hidden
let hiddenIntents = new Set(["background", "uses"]);
let hiddenRoles = new Set();
let pickedLine = "";
let focusId = "";

const linesOf = () => (run.lineages || []).flatMap((call, c) => (call.lines || []).map((line, i) => ({ id: `${c + 1}.${i + 1}`, papers: line.papers.map((p) => p.arxiv_id) })));

function timelineTab() {
  const g = run.graph;
  if (!g || !(g.nodes || []).length) return '<div class="prose"><p class="dim">This run has no citation graph.</p></div>';
  const tally = (i) => g.edges.filter((e) => e.intent === i).length;
  const intents = Object.keys(INTENT_COLORS).filter(tally);
  const roles = Object.keys(ROLES).filter((r) => g.nodes.some((n) => n.role === r));
  const lines = linesOf();
  const chip = (attr, key, on, color, text) => `<button type="button" class="gchip${on ? "" : " off"}" ${attr}="${key}"><i style="background:var(${color})"></i>${text}</button>`;
  return `<div class="cgraph"><h2>Citation graph</h2>
    <p class="dim">Every paper a tool returned in this run, ${g.nodes.length} in all, and the ${g.edges.length.toLocaleString()} citations among them. Older papers sit left and more cited papers sit higher.</p>
    <div class="gpanel">
      <div class="gfield"><span class="st-h">Papers</span>${roles.map((r) => chip("data-r", r, !hiddenRoles.has(r), ROLES[r], `${ROLE_WORDS[r]} <span class="dim">${g.nodes.filter((n) => n.role === r).length}</span>`)).join("")}</div>
      <div class="gfield"><span class="st-h">Citations</span>${intents.map((i) => chip("data-i", i, !hiddenIntents.has(i), INTENT_COLORS[i], `${i} <span class="dim">${tally(i)}</span>`)).join("")}</div>
      ${lines.length ? `<div class="gfield"><span class="st-h">Lines</span><select id="gline"><option value="">none</option>${lines.map((l) => `<option value="${l.id}"${l.id === pickedLine ? " selected" : ""}>Line ${l.id}, ${l.papers.length} papers</option>`).join("")}</select></div>` : ""}
      <div class="gfield"><span class="st-h">Find</span><input type="search" id="gq" placeholder="A title or an arXiv id" value="${esc(focusId)}"></div>
    </div>
    <div class="gwrap" id="gwrap"></div></div>`;
}

function drawGraph() {
  const host = document.getElementById("gwrap");
  const g = run.graph;
  if (!host || !g) return;
  const line = linesOf().find((l) => l.id === pickedLine);
  const onLine = new Set(line ? line.papers : []);
  const shown = g.nodes.filter((n) => !hiddenRoles.has(n.role) || onLine.has(n.arxiv_id));
  if (!shown.length) { host.innerHTML = '<p class="dim">Every kind of paper is hidden.</p>'; return; }
  const at = Object.fromEntries(shown.map((n) => [n.arxiv_id, n]));
  const t = shown.map((n) => Date.parse(n.date));
  const t0 = Math.min(...t);
  const t1 = Math.max(Math.max(...t), t0 + 8.64e7);
  const lg = (n) => Math.log10(1 + (n.cited_by || 0));
  const c1 = Math.max(1, ...shown.map(lg));
  const width = host.clientWidth || 1100;
  const height = Math.max(420, Math.min(680, Math.round(width * 0.5)));
  const pad = { l: 58, r: 24, t: 18, b: 44 };
  const X = (n) => pad.l + ((Date.parse(n.date) - t0) / (t1 - t0)) * (width - pad.l - pad.r);
  const Y = (n) => pad.t + (1 - lg(n) / c1) * (height - pad.t - pad.b);
  const grid = [];
  const years = (t1 - t0) / 3.156e10;
  const step = years > 24 ? 5 : years > 10 ? 2 : 1;
  for (let yr = new Date(t0).getFullYear(); yr <= new Date(t1).getFullYear(); yr += step) {
    const x = pad.l + ((Date.parse(`${yr}-01-01`) - t0) / (t1 - t0)) * (width - pad.l - pad.r);
    if (x < pad.l - 1 || x > width - pad.r) continue;
    grid.push(`<line class="ggrid" x1="${x}" y1="${pad.t}" x2="${x}" y2="${height - pad.b}"/><text class="gaxis" x="${x}" y="${height - pad.b + 18}" text-anchor="middle">${yr}</text>`);
  }
  for (const v of [0, 10, 100, 1000, 10000, 100000]) {
    if (Math.log10(1 + v) > c1) break;
    const y = pad.t + (1 - Math.log10(1 + v) / c1) * (height - pad.t - pad.b);
    grid.push(`<line class="ggrid" x1="${pad.l}" y1="${y}" x2="${width - pad.r}" y2="${y}"/><text class="gaxis" x="${pad.l - 8}" y="${y + 4}" text-anchor="end">${v.toLocaleString()}</text>`);
  }
  grid.push(`<text class="gaxis" transform="translate(14 ${height / 2}) rotate(-90)" text-anchor="middle">cited by &rarr;</text>`);
  const edges = g.edges.map((e, i) => ({ ...e, i, inline: onLine.has(e.source) && onLine.has(e.target) }))
    .filter((e) => at[e.source] && at[e.target] && (e.inline || !hiddenIntents.has(e.intent)));
  const edgeSvg = edges.map((e) => {
    const d = `x1="${X(at[e.source])}" y1="${Y(at[e.source])}" x2="${X(at[e.target])}" y2="${Y(at[e.target])}"`;
    return `<g class="gedge${e.inline ? " inline" : ""}" data-e="${e.i}" data-s="${e.source}" data-t="${e.target}"><line style="stroke:var(${INTENT_COLORS[e.intent] || "--overlay1"})" ${d}/><line class="ghit" ${d}/></g>`;
  }).join("");
  const path = line ? line.papers.filter((a) => at[a]).map((a) => `${X(at[a])},${Y(at[a])}`).join(" ") : "";
  const dots = shown.map((n) => `<circle class="gnode${line && !onLine.has(n.arxiv_id) ? " dim" : ""}${onLine.has(n.arxiv_id) ? " online" : ""}" data-id="${n.arxiv_id}" cx="${X(n)}" cy="${Y(n)}" r="${n.role === "found" ? 5.5 : 8}" style="fill:var(${ROLES[n.role]})"/>`).join("");
  host.innerHTML = `<svg width="${width}" height="${height}" id="gsvg" role="img" aria-label="Citation graph">${grid.join("")}<g>${edgeSvg}</g>${path ? `<polyline class="gline" points="${path}"/>` : ""}${dots}</svg>`;
  const byIndex = Object.fromEntries(edges.map((e) => [e.i, e]));
  const svg = document.getElementById("gsvg");
  const nameOf = (a) => esc(clip(deTeX((at[a] || {}).title || a), 70));
  for (const c of svg.querySelectorAll(".gnode")) {
    const n = at[c.dataset.id];
    const html = `<div class="cp-t">${esc(deTeX(n.title))}</div><div class="cp-m">${esc(n.date)} &middot; ${esc(n.arxiv_id)} &middot; ${(n.cited_by || 0).toLocaleString()} citations</div><div class="cp-m">${ROLE_WORDS[n.role]}</div>`;
    c.onmouseenter = (e) => { spot(n.arxiv_id); showTip(e, html); };
    c.onmousemove = (e) => showTip(e);
    c.onmouseleave = () => { unspot(); hideTip(); };
    c.onclick = () => window.open(`https://arxiv.org/abs/${n.arxiv_id}`, "_blank");
  }
  for (const el of svg.querySelectorAll(".gedge")) {
    const e = byIndex[el.dataset.e];
    const html = `<div><b>${nameOf(e.source)}</b></div><div class="tip-k" style="color:var(${INTENT_COLORS[e.intent] || "--overlay1"})">${esc(e.intent)}</div><div><b>${nameOf(e.target)}</b></div>${e.sentence ? `<blockquote>${esc(e.sentence)}</blockquote>` : ""}`;
    el.onmouseenter = (ev) => { spot(e.source, el); showTip(ev, html); };
    el.onmousemove = (ev) => showTip(ev);
    el.onmouseleave = () => { unspot(); hideTip(); };
  }
  if (focusId) {
    const q = focusId.toLowerCase();
    const hit = shown.find((n) => n.arxiv_id.includes(q) || n.title.toLowerCase().includes(q));
    if (hit) spot(hit.arxiv_id);
  }
}

// a paper under the pointer lights up what it cites and what cites it, and dims the rest
function spot(id, only) {
  const svg = document.getElementById("gsvg");
  if (!svg) return;
  const near = new Set([id]);
  for (const g of svg.querySelectorAll(".gedge")) {
    const hit = only ? g === only : g.dataset.s === id || g.dataset.t === id;
    g.classList.toggle("hi", hit);
    if (hit) { near.add(g.dataset.s); near.add(g.dataset.t); }
  }
  for (const c of svg.querySelectorAll(".gnode")) c.classList.toggle("hi", near.has(c.dataset.id));
  svg.classList.add("focus");
}

function unspot() {
  const svg = document.getElementById("gsvg");
  if (!svg) return;
  svg.classList.remove("focus");
  for (const el of svg.querySelectorAll(".hi")) el.classList.remove("hi");
}

function wireGraph() {
  const root = document.querySelector(".cgraph");
  if (!root) return;
  for (const b of root.querySelectorAll("[data-r]")) b.onclick = () => { const r = b.dataset.r; hiddenRoles.has(r) ? hiddenRoles.delete(r) : hiddenRoles.add(r); b.classList.toggle("off"); drawGraph(); };
  for (const b of root.querySelectorAll("[data-i]")) b.onclick = () => { const i = b.dataset.i; hiddenIntents.has(i) ? hiddenIntents.delete(i) : hiddenIntents.add(i); b.classList.toggle("off"); drawGraph(); };
  const sel = document.getElementById("gline");
  if (sel) sel.onchange = () => { pickedLine = sel.value; drawGraph(); };
  document.getElementById("gq").oninput = (e) => { focusId = e.target.value.trim(); drawGraph(); };
  drawGraph();
}

function checkTab() {
  const c = run.check || {};
  const faults = (c.faults || []).map((f) => `<li><b>${esc(f.kind)}</b>${f.where ? ` <code>${esc(f.where)}</code>` : ""}: ${esc(f.line)}</li>`).join("");
  return `<div class="prose"><p><span class="chip ${c.passed ? "ok" : "warn"}">${c.passed ? "passed" : unverified(faultGroups().length)}</span> ${c.markers || 0} citation markers.</p>
    <p class="dim">The check is mechanical: every marker names a paragraph a read returned, every quote sits in that paragraph, every paper cited in the body was read, and every line stating a finding carries a marker or is labeled as inference.</p>
    ${faults ? `<ul>${faults}</ul>` : ""}</div>`;
}

function showRun(tab) {
  const views = { report: reportTab, papers: papersTab, process: processTab, timeline: timelineTab, check: checkTab };
  const tabs = TABS.map((t) => `<button class="${t === tab ? "active" : ""}" onclick="location.hash='#/run/${encodeURIComponent(run.slug)}/${t}'">${t[0].toUpperCase() + t.slice(1)}</button>`).join("");
  const c = run.check || {};
  const quotes = Object.keys(run.evidence || {}).length;
  document.getElementById("view").innerHTML = `
    <header class="runhead"><a class="back" href="#/">&larr; Library</a>
      <h1>${esc(asked(run.question))}</h1>
      <div class="meta"><span class="chip ${c.passed ? "ok" : "warn"}">${c.passed ? "checked" : unverified(faultGroups().length)}</span>
        <span class="chip">${duration(run.seconds)}</span><span class="chip">${Object.keys(run.papers || {}).length} papers read</span>
        <span class="chip">${quotes} quotes</span><span class="byline">${esc(when(run.saved))}</span></div>
    </header>
    <nav class="tabs">${tabs}</nav>
    <div class="runlayout${tab === "report" || tab === "papers" ? "" : " solo"}"><section class="tab-panel active">${views[tab]()}</section>
      <aside id="inspector" class="inspector"></aside></div>`;
  clearInspector();
  wire(document.getElementById("view"));
  wireResults();
  drawTree();
  wireGraph();
}

// -- routing ---------------------------------------------------------------------------------

async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  pinned = null;
  if (parts[0] === "run" && parts[1]) {
    const slug = decodeURIComponent(parts[1]);
    if (!run || run.slug !== slug) {
      resultsOn = "";
      const got = await fetch(`api/runs/${encodeURIComponent(slug)}`);
      if (!got.ok) { document.getElementById("view").innerHTML = '<p class="empty">No such run.</p>'; return; }
      run = await got.json();
    }
    showRun(TABS.includes(parts[2]) ? parts[2] : "report");
    window.scrollTo(0, 0);
    return;
  }
  runs = await (await fetch("api/runs")).json();
  library();
}

theme();
window.addEventListener("hashchange", route);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") clearInspector(); });
route();
