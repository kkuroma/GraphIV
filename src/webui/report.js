// The report tab at two depths: skim reads the answer and one claim per section, read opens every finding.
// It draws the report object save_report keeps, and falls back to the markdown report for older runs.

const legacyReport = reportTab;
const DEPTH_KEY = "graphiv.depth";
let depth = "skim";
try { depth = localStorage.getItem(DEPTH_KEY) || "skim"; } catch (e) { /* a private window forgets */ }
// a link can fix the depth, as it can the theme
if (["skim", "read"].includes(new URLSearchParams(location.search).get("depth"))) depth = new URLSearchParams(location.search).get("depth");
let paperNo = {};
let evTab = "";
let posetOn = "";
let posetWhole = false;
let posetPageWhole = false;
let posetFocus = null;
let posetPicked = "";
const posetIntents = new Set(["extends", "critiques", "compares"]);

const humanize = (s) => String(s ?? "").replace(/_/g, " ");
const yearOf = (d) => String(d || "").slice(0, 4);
// the title index walks every reply, so it is built once a run
let titleCache = { slug: null, map: {} };
const titleOf = (id) => {
  if (titleCache.slug !== run.slug) titleCache = { slug: run.slug, map: titles() };
  return deTeX(titleCache.map[id] || id);
};

function citesOf(b) {
  const listed = Array.isArray(b.cites) ? b.cites : [];
  const inline = [...String(b.text || "").matchAll(MARKER)].flatMap((m) => m[1].split(",").map((x) => x.trim()));
  return [...new Set([...listed, ...inline])];
}
const bareText = (t) => String(t || "").replace(/\s*\[[A-Z][A-Za-z]*\d+(?:\s*,\s*[A-Z][A-Za-z]*\d+)*\]/g, "").trim();
const paperOfLabel = (l) => ((run.evidence || {})[l] || {}).arxiv_id;

function chipFor(label) {
  const id = paperOfLabel(label);
  if (!id) return "";
  return `<button class="cite" type="button" data-m="${esc(label)}" title="${esc(titleOf(id))}">${paperNo[id] ?? "?"}</button>`;
}

// one chip per paper in a statement, the first of its sentences standing for the rest
function perPaper(labels) {
  const seen = new Set();
  return labels.filter((l) => { const id = paperOfLabel(l); if (!id || seen.has(id)) return false; seen.add(id); return true; });
}

// a title's head before its colon is the paper's name when it reads as one, as save_report decides
function nameOfTitle(t) {
  const s = String(t || "").replace(/\s+/g, " ");
  const i = s.indexOf(":");
  if (i < 0) return "";
  const head = s.slice(0, i).trim();
  return head.split(" ").length <= 3 && head.length >= 3 && /[A-Z]/.test(head) ? head : "";
}
const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function nameOfPaper(id) {
  for (const p of run.lineages || []) for (const q of p.papers || []) if (q.arxiv_id === id && q.name) return q.name;
  return nameOfTitle((run.papers || {})[id]?.title || "");
}

// each chip sits after the name of the paper it quotes, and the rest close the sentence
// markers the writer put inline stay where they are; the others go after the name of their paper, else at the end
function said(text, labels) {
  const raw = proseTeX(String(text || "").replace(/\s+/g, " ").trim()).replace(/\s+([.,;:!?)])/g, "$1");
  const inline = new Set([...raw.matchAll(MARKER)].flatMap((m) => m[1].split(",").map((x) => x.trim())));
  const end = /[.!?]$/.test(raw);
  let body = end ? raw.slice(0, -1) : raw;
  const at = new Map();
  const rest = [];
  for (const l of labels.filter((x) => !inline.has(x))) {
    const name = nameOfPaper(paperOfLabel(l));
    const hit = name ? new RegExp(`(?<![\\w-])${reEsc(name)}(?![\\w-])`).exec(body) : null;
    if (hit) at.set(hit.index + name.length, [...(at.get(hit.index + name.length) || []), l]);
    else rest.push(l);
  }
  for (const spot of [...at.keys()].sort((a, b) => b - a)) body = `${body.slice(0, spot)} [${at.get(spot).join(", ")}]${body.slice(spot)}`;
  if (rest.length) body += ` [${rest.join(", ")}]`;
  // one chip per paper inside each bracket, the first of its sentences standing for the rest
  let out = "";
  let from = 0;
  for (const m of body.matchAll(MARKER)) {
    out += esc(body.slice(from, m.index)) + perPaper(m[1].split(",").map((x) => x.trim())).map(chipFor).join("");
    from = m.index + m[0].length;
  }
  return `${out}${esc(body.slice(from))}${end ? raw.slice(-1) : "."}`;
}

const tableIds = () => Object.fromEntries((run.tables || []).map((t, i) => [`T${i + 1}`, t]));
const posetIds = () => Object.fromEntries((run.lineages || []).filter((p) => p.id).map((p) => [p.id, p]));
const fold = (s) => String(s || "").toLowerCase().split(/\s+/).filter(Boolean).join(" ");
const searchFor = (words) => (run.searches || []).find((s) => fold(s.query) === fold(words));

// papers are numbered in the order the report first leans on them
function numberPapers(o) {
  paperNo = {};
  let next = 1;
  const take = (labels) => labels.forEach((l) => { const id = paperOfLabel(l); if (id && !(id in paperNo)) paperNo[id] = next++; });
  const walk = (b) => {
    if (!b || typeof b !== "object") return;
    take(citesOf(b));
    if (b.type === "table" || b.type === "coverage") (tableIds()[b.table]?.rows || []).forEach((r) => take(Object.values(r.cells).map((c) => c.label).filter(Boolean)));
    if (b.type === "lineage") {
      const p = posetIds()[b.poset];
      const keep = new Set(b.papers?.length ? b.papers : p?.focus || []);
      take((p?.edges || []).filter((e) => keep.has(e.citer) && keep.has(e.cited)).map((e) => e.label).filter(Boolean));
    }
  };
  walk(o.tldr);
  (o.overview || []).forEach(walk);
  (o.sections || []).forEach((s) => (s.blocks || []).forEach(walk));
  for (const t of Object.values(tableIds())) t.rows.forEach((r) => take(Object.values(r.cells).map((c) => c.label).filter(Boolean)));
}

// -- the claims ----------------------------------------------------------------------------

const FIG_WORDS = { table: (b) => `Table ${b.table}`, coverage: (b) => `Coverage of ${b.table}`, lineage: (b) => `How these papers relate, poset ${b.poset}` };
const FIG_TAB = { table: "table", coverage: "coverage", lineage: "poset" };

function openItem(b) {
  const s = searchFor(b.search);
  const near = (s?.papers || []).slice(0, 3).map((p) => `<li>${esc(deTeX(p.title))} ${(run.papers || {})[p.arxiv_id] ? '<span class="rp-badge ok">read</span>' : ""}</li>`).join("");
  const how = s ? `A ${esc(s.mode)} search for <q>${esc(s.query)}</q> returned ${s.papers.length} papers; the nearest:<ul>${near}</ul>` : "Checked by a search this run made.";
  return `<li class="rp-open"><span class="rp-stamp">Open</span>${said(b.text, citesOf(b))}<details class="rp-how"><summary>How it was checked</summary>${how}</details></li>`;
}

function blockItem(b) {
  if (b.type === "finding") return `<li>${said(b.text, citesOf(b))}</li>`;
  if (b.type === "inference") return `<li class="rp-reason"><span class="rp-tag">Reasoning</span>${esc(bareText(b.text))}</li>`;
  if (b.type === "open") return openItem(b);
  if (FIG_WORDS[b.type]) return `<li class="rp-figref"><button type="button" class="pill sm" data-fig="${FIG_TAB[b.type]}" data-poset="${esc(b.poset || "")}" data-papers="${esc((b.papers || []).join(","))}">${esc(FIG_WORDS[b.type](b))} &rarr;</button></li>`;
  return "";
}

function sectionRow(sec, i) {
  const blocks = sec.blocks || [];
  const claims = blocks.filter((b) => b.type === "finding" || b.type === "open");
  const lead = blocks.find((b) => b.type === "finding") || blocks.find((b) => b.type === "open");
  const sources = new Set(blocks.flatMap((b) => citesOf(b)).map(paperOfLabel).filter(Boolean));
  const opens = blocks.filter((b) => b.type === "open").length;
  const figs = blocks.filter((b) => FIG_WORDS[b.type]).length;
  const facts = [
    claims.length ? `${claims.length} claim${claims.length === 1 ? "" : "s"}` : "",
    sources.size ? `${sources.size} source${sources.size === 1 ? "" : "s"}` : "",
    opens ? `${opens} open` : "",
    figs ? `${figs} figure${figs === 1 ? "" : "s"}` : "",
  ].filter(Boolean).join(" · ");
  const head = lead ? `<span class="rp-lead">${esc(proseTeX(bareText(lead.text)))}</span>` : "";
  return `<details class="rp-sec" id="s${i}" ${depth === "read" ? "open" : ""}>
    <summary><span class="rp-h"><b>${esc(sec.heading)}</b><span class="rp-facts">${facts}</span></span>${head}</summary>
    <ul class="rp-claims">${blocks.map(blockItem).join("")}</ul>
    <a class="rp-link" href="#/run/${encodeURIComponent(run.slug)}/report/s${i}" title="Link to this section">#</a></details>`;
}

// -- the evidence card ---------------------------------------------------------------------

function covCard(t, spec) {
  if (!t) return '<p class="dim">This run kept no table.</p>';
  const cols = t.schema.columns;
  const by = cols.find((c) => c.name === spec?.rows) || cols.find((c) => c.kind === "choice");
  let heads = (spec?.columns || cols.filter((c) => c.kind === "bool").map((c) => c.name));
  if (!by || !heads.length) return '<p class="dim">The table has no field to count papers by.</p>';
  const one = heads.length === 1 && cols.find((c) => c.name === heads[0])?.kind === "choice";
  const keys = one ? cols.find((c) => c.name === heads[0]).options : heads;
  const hit = (r, h) => (one ? r.cells[heads[0]]?.value === h : r.cells[h]?.value === "true");
  const lab = (r, h) => (one ? r.cells[heads[0]]?.label : r.cells[h]?.label);
  const groups = by.options.map((o) => [o, t.rows.filter((r) => r.cells[by.name]?.value === o)]).filter(([, m]) => m.length);
  const loose = t.rows.filter((r) => !by.options.includes(r.cells[by.name]?.value));
  if (loose.length) groups.push(["not stated", loose]);
  const cells = [`<div class="h"></div>`, ...keys.map((h) => `<div class="h">${esc(humanize(h))}</div>`)];
  for (const [o, members] of groups) {
    cells.push(`<div class="rl">${esc(humanize(o))}<small>${members.length}</small></div>`);
    for (const h of keys) {
      const got = members.filter((r) => hit(r, h));
      if (!got.length) { cells.push(`<div class="rp-cell zero" title="No paper here states it">none</div>`); continue; }
      const w = got.length / members.length;
      const labels = got.map((r) => lab(r, h)).filter(Boolean);
      cells.push(`<button type="button" class="rp-cell${w > 0.55 ? " hot" : ""}" style="--w:${w.toFixed(2)}" data-labels="${esc(labels.join(","))}" title="${got.map((r) => esc(titleOf(r.arxiv_id))).join("\n")}">${got.length}<small>/${members.length}</small></button>`);
    }
  }
  return `<p class="rp-cap">Papers by ${esc(humanize(by.name))}, counted where their own sentence says yes. Hatched: no paper read here states it. Click a count for its papers.</p>
    <div class="rp-scroll"><div class="rp-cov" style="grid-template-columns: minmax(140px, 1.3fr) repeat(${keys.length}, minmax(76px, 1fr))">${cells.join("")}</div></div>`;
}

// long sentence columns start hidden, so the table fits the page and reads at a glance
let tableShown = null;

function tableCard(t, tid) {
  if (!t) return '<p class="dim">This run kept no table.</p>';
  const all = t.schema.columns;
  if (!tableShown || !all.some((c) => tableShown.has(c.name))) tableShown = new Set(all.filter((c) => c.kind !== "sentences" && c.kind !== "text").map((c) => c.name));
  const cols = all.filter((c) => tableShown.has(c.name));
  // a value opens the sentence behind it, so the table carries no column of citation numbers
  const val = (c, cell) => {
    const v = cell?.value ?? "n/a";
    if (v === "n/a") return `<td class="na">&ndash;</td>`;
    const shown = c.kind === "bool" ? (v === "true" ? '<span class="rp-yes">&check;</span>' : '<span class="na">&times;</span>')
      : c.kind === "choice" ? `<span class="rp-val">${esc(humanize(v))}</span>` : `<span>${esc(proseTeX(v))}</span>`;
    const body = cell?.label ? `<button type="button" class="cite rp-cv" data-m="${esc(cell.label)}">${shown}</button>` : shown;
    return `<td class="${c.kind === "float" ? "num" : c.kind === "text" || c.kind === "sentences" ? "long" : ""}">${body}</td>`;
  };
  const head = cols.map((c) => `<th class="${c.kind}" title="${esc(c.description)}">${esc(humanize(c.name))}</th>`).join("");
  const short = (id) => { const p = paperOf(id); const n = nameOfTitle(p?.title || ""); return n || titleOf(id); };
  const rows = t.rows.map((r) => `<tr><td class="pp" title="${esc(titleOf(r.arxiv_id))}"><span>${esc(short(r.arxiv_id))}</span><small>${yearOf(paperOf(r.arxiv_id)?.date)}</small></td>${cols.map((c) => val(c, r.cells[c.name])).join("")}</tr>`).join("");
  const picks = all.map((c) => `<button type="button" class="gchip${tableShown.has(c.name) ? "" : " off"}" data-col="${esc(c.name)}" title="${esc(c.description)}"><i></i>${esc(humanize(c.name))}</button>`).join("");
  return `<p class="rp-cap">${t.rows.length} papers read in full, one row each; every answer comes from the paper's own text and opens its sentence on click. &times; means no, &ndash; means the paper does not say.</p>
    <div class="rp-picks rp-cols"><span class="dim">Columns</span>${picks}</div>
    <div class="rp-scroll"><table class="rp-table"><thead><tr><th>Paper</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

// -- the poset: papers in half-year columns, each citation an arrow colored by its intent --------

function posetLayout(papers, edges, room) {
  const half = (p) => `${p.date.slice(0, 4)} ${+p.date.slice(5, 7) <= 6 ? "H1" : "H2"}`;
  const keys = [...new Set(papers.map(half))].sort();
  const cols = keys.map((k) => papers.filter((p) => half(p) === k).sort((a, b) => a.date.localeCompare(b.date)));
  const near = new Map(papers.map((p) => [p.arxiv_id, []]));
  edges.forEach((e) => { near.get(e.citer).push(e.cited); near.get(e.cited).push(e.citer); });
  const pos = new Map();
  const place = () => cols.forEach((c) => c.forEach((p, i) => pos.set(p.arxiv_id, i - c.length / 2)));
  place();
  // a few sweeps ordering each column by where its neighbors sit cut most crossings
  for (let sweep = 0; sweep < 8; sweep++) {
    for (const c of cols) {
      const bary = (p) => { const v = near.get(p.arxiv_id).map((b) => pos.get(b)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : pos.get(p.arxiv_id); };
      const by = new Map(c.map((p) => [p.arxiv_id, bary(p)]));
      c.sort((a, b) => by.get(a.arxiv_id) - by.get(b.arxiv_id));
      c.forEach((p, i) => pos.set(p.arxiv_id, i - c.length / 2));
    }
  }
  // columns share the card's width, wide enough for a short name, and scroll only past that
  const COL = Math.max(150, Math.min(230, Math.floor(room / keys.length))), ROW = 30, TOP = 34;
  const tall = Math.max(...cols.map((c) => c.length));
  const xy = new Map();
  cols.forEach((c, j) => c.forEach((p, i) => xy.set(p.arxiv_id, [16 + j * COL, TOP + i * ROW + ROW / 2])));
  return { keys, xy, width: 16 + keys.length * COL + 24, height: TOP + tall * ROW + 12, COL };
}

function posetCard(placed, page) {
  const all = posetIds();
  const ids = Object.keys(all);
  if (!ids.length) return '<p class="dim">This run mapped no poset.</p>';
  const on = posetOn && all[posetOn] ? posetOn : placed[0]?.poset || ids[0];
  const P = all[on];
  const chosen = posetFocus || placed.find((b) => b.poset === on && b.papers?.length)?.papers;
  const focus = new Set(chosen || P.focus || []);
  // papers citing or cited by two of the chosen ones join them, so the drawing connects
  if (chosen) {
    const touch = new Map();
    for (const e of P.edges) for (const [a, b] of [[e.citer, e.cited], [e.cited, e.citer]]) if (focus.has(b) && !focus.has(a)) touch.set(a, (touch.get(a) || 0) + 1);
    [...touch].filter(([, n]) => n >= 2).sort((x, y) => y[1] - x[1]).slice(0, 6).forEach(([a]) => focus.add(a));
  }
  const byId = new Map(P.papers.map((p) => [p.arxiv_id, p]));
  const typed = P.edges.filter((e) => posetIntents.has(e.intent) && byId.has(e.citer) && byId.has(e.cited));
  const whole = page ? posetPageWhole : posetWhole;
  const inView = (e) => whole || (focus.has(e.citer) && focus.has(e.cited));
  const edges = typed.filter((e) => !e.implied && inView(e));
  const used = new Set(whole ? edges.flatMap((e) => [e.citer, e.cited]) : [...focus].filter((a) => byId.has(a)));
  const papers = P.papers.filter((p) => used.has(p.arxiv_id));
  if (!papers.length) return '<p class="dim">No paper of this poset is left under these filters.</p>';
  const into = new Map();
  typed.forEach((e) => into.set(e.cited, (into.get(e.cited) || 0) + 1));
  // the timeline page has the width beside the panel, the report card its own column
  const room = page ? Math.min(innerWidth - 28, 1400) - 40 : (document.querySelector(".rp-ev")?.clientWidth || 1000) - 40;
  const { keys, xy, width, height, COL } = posetLayout(papers, edges, room);
  const r = (id) => 4 + Math.sqrt(into.get(id) || 0) * 1.7;
  const fit = Math.max(14, Math.floor((COL - 26) / 7));
  const label = (p) => { const t = p.name || deTeX(p.title); return t.length > fit ? t.slice(0, fit - 1) + "…" : t; };
  const recent = (p) => p.date >= (P.since || "");
  const axis = keys.map((k, j) => `<text class="pz-col" x="${16 + j * COL - 4}" y="16">${k}</text><line class="pz-rule" x1="${16 + j * COL - 10}" x2="${16 + j * COL - 10}" y1="22" y2="${height}"/>`).join("");
  const arrows = edges.map((e) => {
    const [x1, y1] = xy.get(e.citer), [x2, y2] = xy.get(e.cited);
    const tx = x2 + (x1 >= x2 ? r(e.cited) + 2 : -r(e.cited) - 2), mx = (x1 + tx) / 2;
    const tip = `${label(byId.get(e.citer))} ${e.intent} ${label(byId.get(e.cited))}: ${e.sentence}`;
    return `<path class="pz-e" data-a="${esc(e.citer)}" data-b="${esc(e.cited)}" d="M${x1},${y1}C${mx},${y1} ${mx},${y2} ${tx},${y2}" style="--c:var(${INTENT_COLORS[e.intent]})" marker-end="url(#pz-${e.intent})"><title>${esc(tip)}</title></path>`;
  }).join("");
  const nodes = papers.map((p) => {
    const [x, y] = xy.get(p.arxiv_id);
    const read = (run.papers || {})[p.arxiv_id];
    return `<g class="pz-n${recent(p) ? " now" : ""}${read ? " read" : ""}${posetPicked === p.arxiv_id ? " on" : ""}" data-node="${esc(p.arxiv_id)}" transform="translate(${x},${y})"><circle r="${r(p.arxiv_id).toFixed(1)}"/><text x="${(r(p.arxiv_id) + 4).toFixed(1)}" dy="0.35em">${esc(label(p))}</text><title>${esc(deTeX(p.title))} (${p.date.slice(0, 7)}), ${into.get(p.arxiv_id) || 0} later papers take a position on it</title></g>`;
  }).join("");
  const marks = ["extends", "critiques", "compares"].map((i) => `<marker id="pz-${i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" style="fill:var(${INTENT_COLORS[i]})"/></marker>`).join("");
  const pick = ids.length > 1 ? ids.map((id) => `<button type="button" class="pill sm${id === on ? " on" : ""}" data-pick="${id}">Poset ${id}</button>`).join("") : "";
  const chips = ["extends", "critiques", "compares"].map((i) => `<button type="button" class="gchip${posetIntents.has(i) ? "" : " off"}" data-pzi="${i}"><i style="background:var(${INTENT_COLORS[i]})"></i>${i} ${typed.filter((e) => e.intent === i).length}</button>`).join("");
  const scope = `<button type="button" class="pill sm${whole ? "" : " on"}" data-pzw="0">${focus.size} papers ${chosen ? "the report discusses" : "closest to the question"}</button><button type="button" class="pill sm${whole ? " on" : ""}" data-pzw="1">Whole poset, ${P.papers.length} papers</button>`;
  const legend = `<span class="pz-legend"><span><i class="pz-k read"></i>read in full</span><span><i class="pz-k"></i>not read</span><span><i class="pz-k now"></i>last two years</span><span class="dim">older on the left · click a paper</span></span>`;
  return `<div class="pz-bar"><div class="rp-picks">${pick}${scope}</div><div class="rp-picks">${chips}</div>${legend}</div>
    <div class="rp-scroll pz-wrap"><svg class="pz" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><defs>${marks}</defs>${axis}${arrows}${nodes}</svg></div>`;
}

// a paper's own panel: what later papers in the poset say of it, and what it says of earlier ones
function posetPanel(id) {
  const P = posetIds()[posetOn] || Object.values(posetIds())[0];
  const p = P?.papers.find((q) => q.arxiv_id === id);
  const box = document.getElementById("inspector");
  if (!p || !box) return;
  const of = (a) => P.papers.find((x) => x.arxiv_id === a);
  const nameOf = (a) => { const q = of(a); return q ? q.name || deTeX(q.title) : a; };
  const row = (e, other) => `<div class="pz-row" style="--c:var(${INTENT_COLORS[e.intent]})"><div class="pz-rh"><span class="pz-tag">${esc(e.intent)}</span><span class="pz-who" title="${esc(deTeX(of(other)?.title || other))}">${esc(nameOf(other))}</span><span class="dim">${esc((of(other)?.date || "").slice(0, 4))}</span>${e.label ? chipFor(e.label) : ""}</div><p>${esc(proseTeX(e.sentence || ""))}</p></div>`;
  // critiques first, then extends, then compares, the newest of each first
  const order = { critiques: 0, extends: 1, compares: 2 };
  const by = (k) => (a, b) => (order[a.intent] - order[b.intent]) || String(of(b[k])?.date).localeCompare(String(of(a[k])?.date));
  const ins = P.edges.filter((e) => e.cited === id).sort(by("citer"));
  const outs = P.edges.filter((e) => e.citer === id).sort(by("cited"));
  const tally = (list) => ["critiques", "extends", "compares"].map((i) => [i, list.filter((e) => e.intent === i).length]).filter(([, n]) => n).map(([i, n]) => `${n} ${i}`).join(", ");
  box.innerHTML = `<div class="pz-head"><h3>${esc(deTeX(p.title))}</h3><p class="dim">${esc(p.date.slice(0, 7))} · <a href="https://arxiv.org/abs/${esc(id)}" target="_blank">arXiv ${esc(id)}</a> · ${(run.papers || {})[id] ? "read in full" : "not read in this run"}</p></div>
    <div class="pz-cols"><section><h4>Later papers on it <span class="dim">${ins.length ? tally(ins) : "none"}</span></h4><div class="pz-list">${ins.map((e) => row(e, e.citer)).join("")}</div></section>
    <section><h4>What it says of earlier work <span class="dim">${outs.length ? tally(outs) : "none"}</span></h4><div class="pz-list">${outs.map((e) => row(e, e.cited)).join("")}</div></section></div>`;
  box.classList.add("on");
  pinned = "poset";
  wire(box);
}

function openCard(o) {
  const opens = (o.sections || []).flatMap((s) => (s.blocks || []).filter((b) => b.type === "open"));
  if (!opens.length) return '<p class="dim">The report names no open question.</p>';
  return `<ul class="rp-claims">${opens.map(openItem).join("")}</ul>`;
}

function evidenceBox(o) {
  const blocks = (o.sections || []).flatMap((s) => s.blocks || []);
  const cov = blocks.find((b) => b.type === "coverage");
  const tid = cov?.table || blocks.find((b) => b.type === "table")?.table || Object.keys(tableIds())[0];
  const placed = blocks.filter((b) => b.type === "lineage");
  const opens = blocks.filter((b) => b.type === "open").length;
  const tabs = [
    ["poset", "How the papers relate", Object.keys(posetIds()).length > 0],
    ["coverage", "Coverage", !!tid],
    ["table", `Table${tid ? " " + tid : ""}`, !!tid],
    ["open", `Open questions ${opens}`, opens > 0],
  ].filter(([, , on]) => on);
  if (!tabs.length) return "";
  if (!tabs.some(([k]) => k === evTab)) evTab = tabs[0][0];
  const t = tableIds()[tid];
  const body = { coverage: () => covCard(t, cov), table: () => tableCard(t, tid), poset: () => posetCard(placed), open: () => openCard(o) }[evTab]();
  return `<section class="rp-ev" id="evidence"><div class="rp-evtabs" role="tablist">${tabs.map(([k, name]) => `<button type="button" role="tab" aria-selected="${k === evTab}" data-ev="${k}">${esc(name)}</button>`).join("")}</div><div class="rp-evbody">${body}</div></section>`;
}

// -- the tab -------------------------------------------------------------------------------

function objectReport() {
  const o = run.object;
  if (!o || !Array.isArray(o.sections)) return legacyReport();
  const part = location.hash.split("/")[4] || "";
  if (part.startsWith("ev-")) evTab = part.slice(3);
  numberPapers(o);
  const calls = run.calls || [];
  const count = (tool) => calls.filter((c) => c.tool === tool).length;
  const passed = (run.check || {}).passed;
  const secs = o.sections.filter((s) => (s.blocks || []).some((b) => b.type === "finding" || b.type === "open" || b.type === "inference"));
  const idx = (s) => o.sections.indexOf(s);
  const overview = (o.overview || []).filter((b) => b.type === "finding");
  const outline = [`<a href="#answer" data-to="answer">Answer</a>`, ...secs.map((s) => `<a href="#s${idx(s)}" data-to="s${idx(s)}">${esc(s.heading)}</a>`), `<a href="#evidence" data-to="evidence">Evidence</a>`].join("");
  return `<div class="rp"><div class="rp-top">
    <nav class="rp-outline" aria-label="On this page">${outline}</nav>
    <div class="rp-main">
      <section class="rp-answer" id="answer"><div class="rp-k">The answer</div><p class="rp-tldr">${said(o.tldr?.text, citesOf(o.tldr || {}))}</p>
        ${overview.length ? `<details class="rp-more" ${depth === "read" ? "open" : ""}><summary>Overview, ${overview.length} points</summary><ul class="rp-claims">${(o.overview || []).map(blockItem).join("")}</ul></details>` : ""}
        <div class="rp-made">${Object.keys(run.papers || {}).length} papers read in full, ${(run.searches || []).length} searches, ${duration(run.seconds)}. ${passed ? `<span class="rp-badge ok">every citation checked</span>` : `<span class="rp-badge warn">${unverified(faultGroups().length)}</span>`} <a href="#/run/${encodeURIComponent(run.slug)}/process">How it was made</a></div>
      </section>
      <div class="rp-bar"><button type="button" class="pill sm" data-expand>${depth === "read" ? "Collapse all" : "Expand all"}</button>
        <span class="dim">${secs.length} sections · click a row to open it · j and k to move</span></div>
      <div class="rp-secs">${secs.map((s) => sectionRow(s, idx(s))).join("")}</div>
    </div></div>
    ${evidenceBox(o)}</div>`;
}

function setDepth(d) {
  depth = d;
  try { localStorage.setItem(DEPTH_KEY, d); } catch (e) { /* as above */ }
  for (const el of document.querySelectorAll(".rp-sec, .rp-more")) el.open = d === "read";
  const button = document.querySelector("[data-expand]");
  if (button) button.textContent = d === "read" ? "Collapse all" : "Expand all";
}

function redrawEvidence() {
  const box = document.getElementById("evidence");
  if (!box || !run.object) return;
  box.outerHTML = evidenceBox(run.object);
  wireReport(true);
}

function showEvidence(tab, poset, papers) {
  evTab = tab;
  if (poset) { posetOn = poset; posetWhole = false; posetFocus = papers ? papers.split(",").filter(Boolean) : null; }
  redrawEvidence();
  document.getElementById("evidence")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function wireReport(only) {
  const root = document.querySelector(".rp");
  if (!root) return;
  wire(root);
  for (const b of root.querySelectorAll("[data-ev]")) b.onclick = () => { evTab = b.dataset.ev; redrawEvidence(); };
  wirePoset(root, redrawEvidence, false);
  for (const b of root.querySelectorAll("[data-col]")) b.onclick = () => { const c = b.dataset.col; tableShown.has(c) ? tableShown.delete(c) : tableShown.add(c); redrawEvidence(); };
  for (const b of root.querySelectorAll(".rp-cell[data-labels]")) b.onclick = () => {
    const labels = b.dataset.labels.split(",").filter(Boolean);
    const box = document.getElementById("inspector");
    if (!box) return;
    box.innerHTML = `<div class="ins-label">${labels.length} papers in this cell</div>${labels.map((l) => `<div class="ins-sec"><div class="cp-t">${esc(titleOf(paperOfLabel(l)))} ${chipFor(l)}</div><p class="cp-b">${esc(proseTeX((run.evidence[l] || {}).quote || ""))}</p></div>`).join("")}`;
    box.classList.add("on");
    pinned = "cell";
    wire(box);
  };
  if (only) return;
  const expand = root.querySelector("[data-expand]");
  if (expand) expand.onclick = () => setDepth(depth === "read" ? "skim" : "read");
  // the question folds to two lines, and a click shows all of it
  const q = document.querySelector(".runhead h1");
  if (q && !q.dataset.wired) { q.dataset.wired = "1"; q.classList.add("rp-q"); q.title = "Click to show the whole question"; q.onclick = () => q.classList.toggle("whole"); }
  for (const b of root.querySelectorAll("[data-fig]")) b.onclick = () => showEvidence(b.dataset.fig, b.dataset.poset, b.dataset.papers);
  for (const a of root.querySelectorAll(".rp-outline a")) a.onclick = (e) => {
    e.preventDefault();
    const el = document.getElementById(a.dataset.to);
    if (el?.tagName === "DETAILS") el.open = true;
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  // the outline marks the part in view
  const marks = [...root.querySelectorAll(".rp-outline a")];
  const watch = new IntersectionObserver((seen) => {
    for (const s of seen) if (s.isIntersecting) for (const a of marks) a.classList.toggle("on", a.dataset.to === s.target.id);
  }, { rootMargin: "-20% 0px -70% 0px" });
  for (const a of marks) { const el = document.getElementById(a.dataset.to); if (el) watch.observe(el); }
  const part = (location.hash.split("/")[4] || "").replace(/^ev-.*/, "evidence");
  if (part) { const el = document.getElementById(part); if (el) { if (el.tagName === "DETAILS") el.open = true; setTimeout(() => el.scrollIntoView({ block: "start" }), 50); } }
}

function wirePoset(root, redraw, page) {
  for (const b of root.querySelectorAll("[data-pick]")) b.onclick = () => { posetOn = b.dataset.pick; posetFocus = null; redraw(); };
  for (const b of root.querySelectorAll("[data-pzi]")) b.onclick = () => { const i = b.dataset.pzi; posetIntents.has(i) ? posetIntents.delete(i) : posetIntents.add(i); redraw(); };
  for (const b of root.querySelectorAll("[data-pzw]")) b.onclick = () => { if (page) posetPageWhole = b.dataset.pzw === "1"; else posetWhole = b.dataset.pzw === "1"; redraw(); };
  for (const g of root.querySelectorAll(".pz-n")) g.onclick = () => {
    posetPicked = g.dataset.node;
    // the picked paper's ancestors and descendants stay lit, the rest of the poset fades
    const svg = g.closest("svg");
    const paths = [...svg.querySelectorAll(".pz-e")];
    const walk = (from, a, b) => { const seen = new Set([from]); const st = [from]; while (st.length) { const x = st.pop(); for (const p of paths) if (p.dataset[a] === x && !seen.has(p.dataset[b])) { seen.add(p.dataset[b]); st.push(p.dataset[b]); } } return seen; };
    const keep = new Set([...walk(posetPicked, "a", "b"), ...walk(posetPicked, "b", "a")]);
    for (const n of svg.querySelectorAll(".pz-n")) { n.classList.toggle("fade", !keep.has(n.dataset.node)); n.classList.toggle("on", n.dataset.node === posetPicked); }
    for (const p of paths) p.classList.toggle("fade", !(keep.has(p.dataset.a) && keep.has(p.dataset.b)));
    posetPanel(posetPicked);
    // on the poset page the poset rises to the top so the paper's panel shows under it
    if (page) root.querySelector(".pz-wrap")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
}

reportTab = objectReport;

// the timeline tab is the poset's own page when the run mapped one, the citation graph below it
const legacyTimeline = timelineTab;
timelineTab = function () {
  if (!Object.keys(posetIds()).length) return legacyTimeline();
  const P = posetIds()[posetOn] || Object.values(posetIds())[0];
  const mix = ["extends", "critiques", "compares"].map((i) => `${P.edges.filter((e) => e.intent === i).length} ${i}`).join(", ");
  return `<div class="pz-page"><h2>How the papers relate</h2>
    <p class="dim">Poset ${esc(P.id)} for “${esc(P.query || "")}”: ${P.papers.length} papers and ${P.edges.length} citations with an intent (${mix}), from a survey search, a focused search since ${esc(P.since || "")}, the papers read and the papers arguing with them.</p>
    <div id="pz-page-body">${posetCard([], true)}</div></div>`;
};

function redrawPage() {
  const box = document.getElementById("pz-page-body");
  if (!box) return;
  box.innerHTML = posetCard([], true);
  wirePoset(box, redrawPage, true);
}

const legacyShow = showRun;
showRun = function (tab) {
  legacyShow(tab);
  if (tab === "report") {
    wireReport();
    const first = (run.object && citesOf(run.object.tldr || {})[0]) || null;
    if (first && window.matchMedia("(min-width: 1101px)").matches) inspect(first);
  }
};

// j and k walk the sections, Enter opens one, as a reader of a long list expects
document.addEventListener("keydown", (e) => {
  if (/input|textarea|select/i.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  const rows = [...document.querySelectorAll(".rp-sec > summary")];
  if (!rows.length || !["j", "k"].includes(e.key)) return;
  const at = rows.indexOf(document.activeElement);
  const to = rows[Math.max(0, Math.min(rows.length - 1, at + (e.key === "j" ? 1 : -1)))] || rows[0];
  e.preventDefault();
  to.focus();
  to.scrollIntoView({ block: "nearest", behavior: "smooth" });
});

// the panel leads with the sentence itself and keeps its paragraph one click away
const CONTEXT = 220;
evidenceCard = function (label) {
  const ev = (run.evidence || {})[label];
  if (!ev) return `<p class="cp-s">No stored quote for ${esc(label)}.</p>`;
  const id = ev.arxiv_id;
  const p = paperOf(id);
  const head = `<div class="ins-sec"><div class="cp-t">${esc(titleOf(id))}</div><div class="cp-m">${p ? `${esc(authors(p.authors))} &middot; ${esc(yearOf(p.date))} &middot; ` : ""}<a href="https://arxiv.org/abs/${id}" target="_blank">arXiv ${id}</a>${p ? ` &middot; <a href="pdf/${id}" target="_blank">pdf</a>` : ""}</div></div>`;
  const citing = ev.pid.startsWith("c-");
  const where = citing ? `A sentence in this paper citing ${esc(titleOf(ev.pid.slice(2)))}` : `${esc(ev.section || "Body")}, paragraph ${esc(ev.pid.slice(1))}`;
  if (ev.text.includes("\n") && ev.text.includes(" | ")) return head + `<div class="ins-sec"><div class="cp-s">${where}</div>${chunkTable(ev.text, ev.quote)}</div>`;
  const text = proseTeX(ev.text);
  const quote = proseTeX(ev.quote || "");
  const at = quote ? text.toLowerCase().indexOf(quote.toLowerCase()) : -1;
  let near = `<mark>${esc(quote || text)}</mark>`;
  if (at >= 0) {
    const a = Math.max(0, at - CONTEXT), b = Math.min(text.length, at + quote.length + CONTEXT);
    near = (a ? "&hellip;" : "") + esc(text.slice(a, at)) + `<mark>${esc(text.slice(at, at + quote.length))}</mark>` + esc(text.slice(at + quote.length, b)) + (b < text.length ? "&hellip;" : "");
  }
  const whole = at >= 0 && (at > CONTEXT || text.length - at - quote.length > CONTEXT)
    ? `<details class="ins-whole"><summary>Whole paragraph</summary><p class="cp-b">${esc(text.slice(0, at))}<mark>${esc(text.slice(at, at + quote.length))}</mark>${esc(text.slice(at + quote.length))}</p></details>` : "";
  return head + `<div class="ins-sec"><div class="cp-s">${where}</div><p class="cp-b">${near}</p>${whole}</div>`;
};

// -- papers: every paper the run read or quoted, filtered and sorted the way a reader asks ---------

let paperQuery = "";
let paperShow = "all";
let paperSort = "cited";

function tableRowOf(id) {
  for (const [tid, t] of Object.entries(tableIds())) {
    const r = t.rows.find((x) => x.arxiv_id === id);
    if (r) return [tid, t, r];
  }
  return [null, null, null];
}

function paperCard(id, read) {
  const p = paperOf(id) || {};
  const [tid, t, r] = tableRowOf(id);
  const answers = r ? t.schema.columns.filter((c) => c.kind !== "sentences" && c.kind !== "text").map((c) => {
    const cell = r.cells[c.name] || {};
    if (!cell.value || cell.value === "n/a") return "";
    const v = c.kind === "bool" ? (cell.value === "true" ? `${humanize(c.name)}` : "") : `${humanize(c.name)}: ${humanize(cell.value)}`;
    return v ? `<button type="button" class="cite rp-cv rp-ans" data-m="${esc(cell.label)}">${esc(v)}</button>` : "";
  }).filter(Boolean).join("") : "";
  const kept = Object.entries(run.evidence || {}).filter(([, e]) => e.arxiv_id === id);
  const quotes = kept.map(([l, e]) => `<li><button class="cite" type="button" data-m="${esc(l)}">${paperNo[id] ?? "&para;"}</button> ${esc(proseTeX(e.quote))}</li>`).join("");
  const n = paperNo[id];
  return `<div class="pcard ${n ? "cited" : ""}">
    <div class="ref-t">${n ? `<span class="rp-no">${n}</span>` : ""}<a href="https://arxiv.org/abs/${esc(id)}" target="_blank">${esc(titleOf(id))}</a></div>
    <div class="ref-m">${esc(authors(p.authors))}${p.date ? ` &middot; ${esc(yearOf(p.date))}` : ""} &middot; ${esc(id)}${p.topic ? ` &middot; ${esc(p.topic)}` : ""}</div>
    <div class="rp-pbadges">${n ? '<span class="rp-badge ok">cited in the report</span>' : ""}<span class="rp-badge">${read ? "read in full" : "quoted from a citing paper"}</span>${tid ? `<span class="rp-badge">row in Table ${tid}</span>` : ""}</div>
    ${answers ? `<div class="rp-answers">${answers}</div>` : ""}
    ${quotes ? `<details class="rp-kept"><summary>${kept.length} sentence${kept.length === 1 ? "" : "s"} kept from it</summary><ul>${quotes}</ul></details>` : ""}
  </div>`;
}

function paperList() {
  const read = Object.keys(run.papers || {});
  const quoted = [...new Set(Object.values(run.evidence || {}).map((e) => e.arxiv_id))].filter((a) => !read.includes(a));
  let ids = [...read, ...quoted];
  if (paperShow === "cited") ids = ids.filter((a) => paperNo[a]);
  if (paperShow === "uncited") ids = ids.filter((a) => !paperNo[a]);
  const q = paperQuery.trim().toLowerCase();
  if (q) ids = ids.filter((a) => `${titleOf(a)} ${a} ${(paperOf(a) || {}).authors || ""}`.toLowerCase().includes(q));
  const date = (a) => String((paperOf(a) || {}).date || "");
  ids.sort(paperSort === "newest" ? (a, b) => date(b).localeCompare(date(a)) : paperSort === "oldest" ? (a, b) => date(a).localeCompare(date(b)) : (a, b) => (paperNo[a] || 999) - (paperNo[b] || 999));
  return { ids, read, all: read.length + quoted.length };
}

function drawPapers() {
  const { ids, read } = paperList();
  const box = document.getElementById("rp-plist");
  if (!box) return;
  box.innerHTML = ids.length ? ids.map((a) => paperCard(a, read.includes(a))).join("") : '<p class="dim">No paper matches.</p>';
  wire(box);
  document.getElementById("rp-pcount").textContent = `${ids.length} shown`;
}

papersTab = function () {
  if (run.object) numberPapers(run.object);
  const { all, read } = paperList();
  const cited = Object.keys(paperNo).length;
  return `<div class="rp-papers"><div class="rp-ptools">
      <input type="search" id="rp-pq" placeholder="Filter by title, author or arXiv id" value="${esc(paperQuery)}" aria-label="Filter papers">
      <div class="rp-depth" role="group" aria-label="Which papers">${[["all", `All ${all}`], ["cited", `Cited ${cited}`], ["uncited", "Not cited"]].map(([k, w]) => `<button type="button" data-show="${k}" aria-pressed="${k === paperShow}">${w}</button>`).join("")}</div>
      <select id="rp-psort" class="pill sm" aria-label="Order"><option value="cited"${paperSort === "cited" ? " selected" : ""}>In citation order</option><option value="newest"${paperSort === "newest" ? " selected" : ""}>Newest first</option><option value="oldest"${paperSort === "oldest" ? " selected" : ""}>Oldest first</option></select>
      <span class="dim" id="rp-pcount"></span></div>
    <p class="dim">${read.length} papers read in full, ${all - read.length} more quoted through a paper citing them, ${cited} cited in the report. A paper's answers and sentences open in the panel.</p>
    <div class="pgrid" id="rp-plist"></div></div>`;
};

function wirePapers() {
  const q = document.getElementById("rp-pq");
  if (!q) return;
  q.oninput = () => { paperQuery = q.value; drawPapers(); };
  for (const b of document.querySelectorAll("[data-show]")) b.onclick = () => {
    paperShow = b.dataset.show;
    for (const x of document.querySelectorAll("[data-show]")) x.setAttribute("aria-pressed", String(x === b));
    drawPapers();
  };
  document.getElementById("rp-psort").onchange = (e) => { paperSort = e.target.value; drawPapers(); };
  drawPapers();
}

// -- check: what the check holds a report to, and every save until it passed ----------------------

checkTab = function () {
  const c = run.check || {};
  const saves = (run.calls || []).filter((x) => x.tool === "save_report");
  const rules = [
    "Every citation names a sentence a read or a citation list returned in this run.",
    "Every number in a statement sits in the paragraph behind its citations, and every quoted phrase too.",
    "Every method or dataset named appears in a cited paper or in a paper the run read.",
    "Every open question names a search the run made, and one of that search's three nearest papers was read.",
    "A deep report holds a comparison table, its coverage, a lineage trace and checked open questions, and has read the top papers of the question's own searches.",
  ];
  const attempt = (x, i) => {
    const lines = String(x.reply || "").split("\n");
    const faults = lines.filter((l) => l.startsWith("- ")).map((l) => `<li>${esc(l.slice(2))}</li>`).join("");
    const what = x.args?.report ? "Sent the whole report" : `Fixed ${Object.keys(x.args?.patch || {}).length} field${Object.keys(x.args?.patch || {}).length === 1 ? "" : "s"}`;
    const ok = lines[0].startsWith("PASSED");
    return `<details class="rp-try" ${!ok && i === saves.length - 1 ? "open" : ""}><summary><span class="rp-badge ${ok ? "ok" : "warn"}">${ok ? "passed" : esc(lines[0].replace(/^NOT PASSED: /, "").split(".")[0].replace(/^1 faults$/, "1 fault"))}</span> ${what} <span class="dim">at ${clock(x.start || 0)}</span></summary>${faults ? `<ul>${faults}</ul>` : ""}</details>`;
  };
  const open = (c.faults || []).map((f) => `<li><code>${esc(f.path || "")}</code> ${esc(f.kind)}${f.where ? `: ${esc(f.where)}` : ""}</li>`).join("");
  return `<div class="rp-check">
    <p><span class="chip ${c.passed ? "ok" : "warn"}">${c.passed ? "passed" : unverified(faultGroups().length)}</span> ${c.markers || 0} citations, after ${saves.length} save${saves.length === 1 ? "" : "s"}.</p>
    ${open ? `<h3>Still failing</h3><ul>${open}</ul>` : ""}
    <h3>What the check holds a report to</h3><ul class="rp-rules">${rules.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
    <p class="dim">The check matches strings only: it never asks a model whether a claim is true, it asks whether the paper says it.</p>
    ${saves.length ? `<h3>Every save</h3>${saves.map(attempt).join("")}` : ""}
    <p class="dim">Download: <a href="files/${encodeURIComponent(run.slug)}/report.md">report.md</a> &middot; <a href="files/${encodeURIComponent(run.slug)}/run.json">run.json</a> &middot; <a href="files/${encodeURIComponent(run.slug)}/calls.jsonl">calls.jsonl</a></p></div>`;
};

const shownRun = showRun;
showRun = function (tab) {
  shownRun(tab);
  if (tab === "papers") wirePapers();
  if (tab === "timeline") {
    const box = document.getElementById("pz-page-body");
    if (box) wirePoset(box, redrawPage, true);
    const panel = document.getElementById("inspector");
    if (box && panel) panel.innerHTML = '<p class="dim">Click a paper in the poset to see what later papers say of it and what it says of earlier work, each with its sentence.</p>';
  }
};

// back to the top once the reader has scrolled a screen
(() => {
  const up = document.getElementById("totop");
  if (!up) return;
  addEventListener("scroll", () => up.classList.toggle("on", scrollY > innerHeight), { passive: true });
  up.onclick = () => scrollTo({ top: 0, behavior: "smooth" });
})();
