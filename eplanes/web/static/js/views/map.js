import * as api from "../api.js";
import { addAirports, addTracks, canvasFallback, createMap, fit, highlight, hoverTracks } from "../map.js";
import { acColor, acKey, AIRCRAFT_ORDER, chip, fmt, h, typeLabel } from "../ui.js";
import { baseBounds, shortName } from "./overview.js";

export async function renderMap(app, params) {
  const [tracks, airports, sessions, aircraft] = await Promise.all([api.get("tracks"), api.get("airports"), api.sessions(), api.get("aircraft")]);
  const bases = [...new Set(AIRCRAFT_ORDER.map((l) => (aircraft.find((a) => a.label === l) || {}).home_airfield).filter(Boolean))];
  const byId = new Map(sessions.map((s) => [s.flight_id, s]));
  app.innerHTML = "";
  const acs = AIRCRAFT_ORDER.filter((a) => tracks.some((t) => t.aircraft_label === a));
  const f = { aircraft: params.get("aircraft") ? params.get("aircraft").split(",") : acs, from: params.get("from") || "", to: params.get("to") || "" };

  const wrap = h("div", { class: "wrap" });
  app.append(wrap);
  wrap.append(h("div", { class: "section", style: { paddingBottom: "14px" } },
    h("h1", {}, "Where the aircraft flew"),
    h("p", { class: "muted" }, `${fmt.int(tracks.length)} sessions carry a GPS track. The logger stores latitude and longitude without their signs; positions are placed by matching take-off and landing points to known airfields, and airfields are shown only where a fix lies within 3 km of one.`)));

  const count = h("span", { class: "active-filters" });
  const baseChips = h("div", { class: "chips" });
  const fromIn = h("input", { type: "date", value: f.from, onchange: (e) => { f.from = e.target.value; draw(); } });
  const toIn = h("input", { type: "date", value: f.to, onchange: (e) => { f.to = e.target.value; draw(); } });
  wrap.append(h("div", { class: "filters" },
    h("div", { class: "field" }, "Aircraft", h("div", { class: "chips" }, acs.map((a) => chip(a, f.aircraft.includes(a), (on) => { f.aircraft = on ? [...f.aircraft, a] : f.aircraft.filter((x) => x !== a); draw(); }, acColor(a))))),
    h("div", { class: "field" }, "Date range (UTC)", h("div", { style: { display: "flex", gap: "6px", alignItems: "center" } }, fromIn, h("span", { class: "faint" }, "to"), toIn)),
    h("div", { class: "field" }, "Go to base", baseChips),
    h("div", { class: "field" }, " ", count)));

  const grid = h("div", { class: "grid side", style: { marginTop: "16px" } });
  wrap.append(grid);
  const mapEl = h("div", { class: "mini-map", style: { height: "min(70vh, 640px)" } });
  const info = h("div", {}, h("p", { class: "muted" }, "Point at a track to see the session; click to select it."));
  const apList = h("div");
  grid.append(h("div", { class: "panel", style: { padding: "0", overflow: "hidden" } }, mapEl),
    h("div", { style: { display: "grid", gap: "20px", alignContent: "start" } },
      h("div", { class: "panel" }, h("h3", {}, "Selected session"), info),
      h("div", { class: "panel" }, h("h3", {}, "Airfields matched"), h("p", { class: "sub" }, "Session start or end fixes within 3 km of the airfield reference point."), apList)));

  const legendBox = h("div", { class: "map-legend" }, acs.map((a) => h("div", { class: "row" }, h("span", { class: "sw", style: { background: acColor(a) } }), a)),
    h("div", { class: "row" }, h("span", { class: "sw" }), "Selected"), h("div", { class: "row" }, h("span", { class: "sw ring" }), "Airfield"));
  mapEl.append(legendBox);

  for (const a of [...airports].sort((x, y) => y.fixes - x.fixes)) {
    apList.append(h("div", { style: { display: "flex", justifyContent: "space-between", gap: "10px", padding: "7px 0", borderTop: "1px solid var(--line)" } },
      h("span", {}, h("b", { class: "display" }, a.code), " ", h("span", { class: "muted" }, a.name), a.municipality ? h("span", { class: "faint" }, `, ${a.municipality}`) : null),
      h("span", { class: "num faint" }, `${fmt.int(a.fixes)} fixes`)));
  }
  if (!airports.length) apList.append(h("p", { class: "faint" }, "No session endpoint lies within 3 km of a known airfield."));

  const show = (fid) => {
    const s = byId.get(fid);
    info.innerHTML = "";
    if (!s) return;
    info.append(h("p", {}, acKey(s.aircraft_label), h("br"), h("b", {}, `${typeLabel(s.session_type)}, ${fmt.dateTime(s.start_utc)}`)),
      h("p", { class: "muted" }, `${fmt.dur(s.duration_min)} · ${fmt.num(s.track_km, 1)} km · ${s.dep_airport || "?"} → ${s.arr_airport || "?"}`,
        s.b1_soc_start !== null ? h("br") : null, s.b1_soc_start !== null ? `Pack 1 SOC ${fmt.int(s.b1_soc_start)} → ${fmt.int(s.b1_soc_end)}%` : null),
      h("a", { class: "btn small", href: `#/flights/${fid}` }, "Open session"));
  };

  let map = null, ready = false;
  const visible = () => {
    const from = f.from ? Date.parse(f.from + "T00:00:00Z") : null, to = f.to ? Date.parse(f.to + "T23:59:59Z") : null;
    return tracks.filter((t) => f.aircraft.includes(t.aircraft_label) && (!from || Date.parse(t.start_utc) >= from) && (!to || Date.parse(t.start_utc) <= to));
  };
  function draw() {
    const v = visible();
    count.textContent = `${fmt.int(v.length)} of ${fmt.int(tracks.length)} tracks shown`;
    const qs = new URLSearchParams();
    if (f.aircraft.length !== acs.length) qs.set("aircraft", f.aircraft.join(","));
    if (f.from) qs.set("from", f.from);
    if (f.to) qs.set("to", f.to);
    history.replaceState(null, "", `#/map${qs.toString() ? "?" + qs : ""}`);
    if (!map) { canvasFallback(mapEl, v, airports); mapEl.append(legendBox); return; }
    if (ready) addTracks(map, v, { colorBy: "aircraft", width: 1.5, opacity: 0.7 });
  }
  if (!tracks.length) { mapEl.append(h("div", { class: "map-empty" }, "This dataset has no GPS tracks.")); return; }
  map = createMap(mapEl);
  draw();
  if (!map) return;
  map.on("load", () => {
    ready = true;
    draw();
    addAirports(map, airports);
    const go = (code, animate) => {
      const b = baseBounds(tracks, code, airports);
      if (b) map.fitBounds(b, { padding: 40, duration: animate ? 900 : 0, maxZoom: 11 });
      baseChips.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.code === code)));
    };
    for (const code of bases) {
      const ap = airports.find((a) => a.code === code);
      const b = h("button", { class: "chip", type: "button", "data-code": code }, ap ? `${shortName(ap)} (${code})` : code);
      b.addEventListener("click", () => go(code, true));
      baseChips.append(b);
    }
    go(bases[0] || (airports[0] && airports[0].code), false);
    hoverTracks(map, "tracks", (p) => { if (p) show(p.flight_id); });
    map.on("click", "tracks", (e) => { const fid = e.features[0].properties.flight_id; highlight(map, fid); show(fid); });
  });
}
