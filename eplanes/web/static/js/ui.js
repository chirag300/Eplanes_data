// Small DOM and formatting helpers shared by all views.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Fixed aircraft order -> fixed identity colour (never re-assigned when a filter hides an aircraft).
export const AIRCRAFT_ORDER = ["C-GAUW", "C-GMUW", "VSW1280102", "VSW1280107"];
export function acIndex(label) {
  const i = AIRCRAFT_ORDER.indexOf(label);
  return i < 0 ? 3 : i;
}
export function acColor(label) { return css(`--s${acIndex(label) + 1}`); }
export const AC_SYMBOL = ["circle", "rect", "triangle", "diamond"];
export function acSymbol(label) { return AC_SYMBOL[acIndex(label)]; }
export function acKey(label) {
  const i = acIndex(label);
  return h("span", { class: "ac-key" }, h("i", { class: i === 3 ? "diamond" : "", style: { background: acColor(label) } }), label);
}

export const fmt = {
  int: (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Math.round(v).toLocaleString("en-CA")),
  num: (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString("en-CA", { minimumFractionDigits: d, maximumFractionDigits: d })),
  pct: (v, d = 0) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(d)}%`),
  date: (v) => (v ? new Date(v).toLocaleDateString("en-CA", { timeZone: "UTC", year: "numeric", month: "short", day: "numeric" }) : "–"),
  dateIso: (v) => (v ? new Date(v).toISOString().slice(0, 10) : "–"),
  dateTime: (v) => (v ? new Date(v).toLocaleString("en-CA", { timeZone: "UTC", year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) + " UTC" : "–"),
  time: (v) => (v ? new Date(v).toLocaleTimeString("en-CA", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", hour12: false }) + " UTC" : "–"),
  dur: (min) => {
    if (min === null || min === undefined) return "–";
    const m = Math.round(min);
    return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
  },
  ago: (iso) => {
    if (!iso) return "never";
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${Math.round(s / 3600)} h ago`;
    return `${Math.round(s / 86400)} days ago`;
  },
  bytes: (b) => (b === null || b === undefined ? "–" : b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`),
};

export const TYPE_LABEL = {
  "flight test": "Flight", "charging": "Charge", "ground test": "Ground run",
  "flight test and charging": "Flight + charge", "ground test and charging": "Ground + charge",
};
export const typeLabel = (t) => TYPE_LABEL[t] || (t ? t[0].toUpperCase() + t.slice(1) : "Unknown");
export const isFlight = (t) => typeof t === "string" && t.startsWith("flight test");

const ICONS = {
  serious: '<path d="M12 3 2 21h20L12 3z"/><path d="M12 10v5M12 18h.01"/>',
  warning: '<path d="M12 3 2 21h20L12 3z"/><path d="M12 10v5M12 18h.01"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/>',
  good: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  plane: '<path d="M12 3v16M4 11l8-1.5 8 1.5M9 19l3-.7 3 .7"/>',
  bolt: '<path d="M13 3 6 13h5l-1 8 7-10h-5l1-8z"/>',
  ground: '<circle cx="12" cy="12" r="8"/><path d="M8 12h8"/>',
};
export function icon(name, cls = "") {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("fill", "none");
  s.setAttribute("stroke", "currentColor");
  s.setAttribute("stroke-width", "2");
  s.setAttribute("stroke-linecap", "round");
  s.setAttribute("stroke-linejoin", "round");
  s.setAttribute("aria-hidden", "true");
  if (cls) s.setAttribute("class", cls);
  s.innerHTML = ICONS[name] || "";
  return s;
}

export function callout(level, text) {
  const label = { serious: "Data fault", warning: "Caution", info: "Note", good: "OK" }[level] || "Note";
  return h("div", { class: `callout ${level}`, role: level === "serious" ? "alert" : null }, icon(level), h("div", {}, h("b", {}, label + ". "), text));
}

export function loading(text = "Loading data…") {
  return h("div", { class: "loading", role: "status" }, h("div", { class: "dot busy" }), h("div", {}, text));
}

export function errorBox(err, retry) {
  return h("div", { class: "empty", role: "alert" },
    h("p", {}, h("b", {}, "This view could not be loaded. "), err && err.message ? err.message : String(err)),
    retry ? h("button", { class: "btn", type: "button", onclick: retry }, "Try again") : null);
}

let toastTimer = null;
export function toast(text, ms = 4500) {
  document.querySelectorAll(".toast").forEach((t) => t.remove());
  const t = h("div", { class: "toast", role: "status" }, text);
  document.body.append(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), ms);
}

export function chip(label, pressed, onToggle, color) {
  const b = h("button", { class: "chip", type: "button", "aria-pressed": String(!!pressed) },
    color ? h("span", { class: "key", style: { background: color } }) : null, label);
  b.addEventListener("click", () => {
    const now = b.getAttribute("aria-pressed") !== "true";
    b.setAttribute("aria-pressed", String(now));
    onToggle(now);
  });
  return b;
}

export function copyButton(getText) {
  const b = h("button", { class: "btn small ghost copy", type: "button" }, icon("copy"), "Copy");
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(getText()); b.lastChild.textContent = "Copied"; }
    catch (e) { b.lastChild.textContent = "Copy failed"; }
    setTimeout(() => { b.lastChild.textContent = "Copy"; }, 1600);
  });
  return b;
}

// Sortable, paginated table. columns: [{key, label, num, render(row), sortValue(row)}]
export function dataTable({ columns, rows, pageSize = 50, initialSort = null, onRow = null, rowClass = null, caption = null }) {
  let sortKey = initialSort ? initialSort.key : null;
  let dir = initialSort ? initialSort.dir : "desc";
  let page = 0;
  const wrap = h("div");
  const body = h("tbody");
  const head = h("tr");
  const pager = h("div", { class: "pager" });
  const table = h("table", { class: "data" }, caption ? h("caption", { class: "sr-only" }, caption) : null, h("thead", {}, head), body);
  wrap.append(h("div", { class: "table-wrap" }, table), pager);

  function sorted() {
    if (!sortKey) return rows;
    const col = columns.find((c) => c.key === sortKey);
    const val = col.sortValue || ((r) => r[sortKey]);
    const m = dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const x = val(a), y = val(b);
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (x > y ? 1 : x < y ? -1 : 0) * m;
    });
  }
  function draw() {
    head.innerHTML = "";
    for (const c of columns) {
      const th = h("th", { class: c.num ? "num" : "", scope: "col" });
      if (c.sortable === false) th.append(c.label);
      else {
        th.setAttribute("aria-sort", sortKey === c.key ? (dir === "asc" ? "ascending" : "descending") : "none");
        th.append(h("button", { type: "button", onclick: () => { if (sortKey === c.key) dir = dir === "asc" ? "desc" : "asc"; else { sortKey = c.key; dir = c.num ? "desc" : "asc"; } page = 0; draw(); } }, c.label));
      }
      head.append(th);
    }
    const all = sorted();
    const n = all.length;
    const pages = Math.max(1, Math.ceil(n / pageSize));
    page = Math.min(page, pages - 1);
    body.innerHTML = "";
    if (!n) body.append(h("tr", {}, h("td", { colspan: columns.length, class: "wrap" }, "No sessions match these filters. Widen the date range or clear a filter.")));
    for (const r of all.slice(page * pageSize, (page + 1) * pageSize)) {
      const tr = h("tr", { class: rowClass ? rowClass(r) : "", tabindex: onRow ? "0" : null });
      for (const c of columns) {
        const v = c.render ? c.render(r) : r[c.key];
        tr.append(h("td", { class: (c.num ? "num " : "") + (c.cls || "") }, v === null || v === undefined ? "–" : v));
      }
      if (onRow) {
        tr.addEventListener("click", (e) => { if (!e.target.closest("input,button,a")) onRow(r); });
        tr.addEventListener("keydown", (e) => { if (e.key === "Enter") onRow(r); });
      }
      body.append(tr);
    }
    pager.innerHTML = "";
    pager.hidden = pages === 1 && n <= pageSize && n < 12;
    pager.append(
      h("span", {}, n ? `Rows ${fmt.int(page * pageSize + 1)}–${fmt.int(Math.min(n, (page + 1) * pageSize))} of ${fmt.int(n)}` : "No rows"),
      h("span", {},
        h("button", { class: "btn small", type: "button", disabled: page === 0 ? true : null, onclick: () => { page -= 1; draw(); } }, "Previous"), " ",
        h("span", { class: "faint" }, `Page ${page + 1} of ${pages}`), " ",
        h("button", { class: "btn small", type: "button", disabled: page >= pages - 1 ? true : null, onclick: () => { page += 1; draw(); } }, "Next")));
  }
  draw();
  wrap.update = (newRows) => { rows = newRows; page = 0; draw(); };
  return wrap;
}

export function median(a) {
  const v = a.filter((x) => x !== null && x !== undefined && !Number.isNaN(x)).sort((x, y) => x - y);
  if (!v.length) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
export const sum = (a) => a.reduce((s, x) => s + (x || 0), 0);

export function onThemeChange(cb) {
  const mo = new MutationObserver(cb);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const mq = window.matchMedia("(prefers-color-scheme: light)");
  mq.addEventListener("change", cb);
  return () => { mo.disconnect(); mq.removeEventListener("change", cb); };
}
