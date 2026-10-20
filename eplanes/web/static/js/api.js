// Data access. Live mode talks to the Eplanes server; static mode reads a published JSON snapshot.
// Every request is a real network fetch; refresh() bypasses all caches.

const CFG = window.EPLANES_CONFIG || { mode: "live", api: "api/" };
export const MODE = CFG.mode;

const memo = new Map();
let generation = 0; // bumps on every refresh so stale responses are never shown as current

function url(path) {
  if (MODE === "static") {
    const map = {
      "status": "data/status.json",
      "sessions": "data/sessions.json",
      "tracks": "data/tracks.json",
      "airports": "data/airports.json",
      "aircraft": "data/aircraft.json",
      "quality": "data/quality.json",
      "dictionary": "data/dictionary.json",
      "downloads": "data/downloads.json",
    };
    if (map[path]) return map[path];
    let m = path.match(/^sessions\/(\d+)$/);
    if (m) return `data/sessions/${m[1]}.json`;
    m = path.match(/^sessions\/(\d+)\/series/);
    if (m) return `data/series/${m[1]}.json`;
    throw new Error(`No static file for ${path}`);
  }
  return CFG.api + path;
}

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function fetchJson(path, { method = "GET", fresh = false } = {}) {
  let u = url(path);
  if (fresh || MODE === "static") u += (u.includes("?") ? "&" : "?") + "_=" + Date.now();
  let r;
  try {
    r = await fetch(u, { method, cache: fresh ? "no-store" : "default", headers: { Accept: "application/json" } });
  } catch (e) {
    throw new ApiError("The data service could not be reached. Check that the Eplanes server is running and try again.", 0);
  }
  if (!r.ok) {
    let detail = "";
    try { const j = await r.json(); detail = j.detail || j.message || ""; } catch (e) { /* not json */ }
    if (r.status === 404) throw new ApiError(detail || "Not found.", 404);
    throw new ApiError(detail || `The data service answered ${r.status}.`, r.status);
  }
  return r.json();
}

export async function get(path, opts = {}) {
  const key = generation + "|" + path;
  if (!opts.fresh && memo.has(key)) return memo.get(key);
  const p = fetchJson(path, opts).catch((e) => { memo.delete(key); throw e; });
  memo.set(key, p);
  return p;
}

export function status() { return fetchJson("status", { fresh: true }); }

export async function refresh({ portal = false } = {}) {
  generation += 1;
  memo.clear();
  if (MODE === "static") {
    const st = await fetchJson("status", { fresh: true });
    return { refresh: { ok: true, at: new Date().toISOString(), message: "Re-fetched the published snapshot." }, ...st };
  }
  return fetchJson(`refresh?portal=${portal ? "true" : "false"}`, { method: "POST", fresh: true });
}

// Sessions arrive column-oriented; expose them as objects once and reuse.
let sessionsCache = null;
export async function sessions() {
  const key = generation;
  if (sessionsCache && sessionsCache.key === key) return sessionsCache.rows;
  const p = await get("sessions");
  const rows = p.rows.map((r) => {
    const o = {};
    p.columns.forEach((c, i) => { o[c] = r[i]; });
    o.start = o.start_utc ? new Date(o.start_utc) : null;
    return o;
  });
  sessionsCache = { key, rows };
  return rows;
}

export function series(fid, maxPoints = 2000) {
  return get(`sessions/${fid}/series?max_points=${maxPoints}`);
}

export function downloadCsv(filename, rows, columns) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const text = [columns.join(","), ...rows.map((r) => columns.map((c) => esc(r[c])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
