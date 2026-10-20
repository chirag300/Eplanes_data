// Moving-map helpers (MapLibre GL). Basemap tiles are raster; if they cannot load (offline, blocked)
// the map still draws tracks and airfields on a plain background. Without WebGL a canvas fallback draws the tracks.
import { acColor, css, h } from "./ui.js";

const CFG = window.EPLANES_CONFIG || {};
// Basemap: OpenFreeMap vector styles - free, no API key or registration, commercial use allowed (https://openfreemap.org).
// Override with map_style_dark / map_style_light in config.js (e.g. a self-hosted PMTiles style).
const STYLE = {
  dark: CFG.map_style_dark || "https://tiles.openfreemap.org/styles/dark",
  light: CFG.map_style_light || "https://tiles.openfreemap.org/styles/positron",
};

function isLight() { return getComputedStyle(document.documentElement).colorScheme === "light"; }

// Used when the basemap cannot be reached (offline, blocked): tracks and airfields still draw on a plain background.
function plainStyle() {
  return { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": css("--panel-2") } }] };
}

export function webglOk() {
  try { const c = document.createElement("canvas"); return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl"))); }
  catch (e) { return false; }
}

export function bounds(coordsList) {
  let w = 180, s = 90, e = -180, n = -90;
  for (const cs of coordsList) for (const [x, y] of cs) { if (x < w) w = x; if (x > e) e = x; if (y < s) s = y; if (y > n) n = y; }
  return w > e ? null : [[w, s], [e, n]];
}

export function createMap(el, { interactive = true, center = [-80.38, 43.46], zoom = 8 } = {}) {
  if (!window.maplibregl || !webglOk()) return null;
  const map = new window.maplibregl.Map({ container: el, style: isLight() ? STYLE.light : STYLE.dark, center, zoom, interactive,
    attributionControl: { compact: true }, cooperativeGestures: false });
  if (interactive) map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  let fellBack = false;
  const fallBack = () => {
    if (fellBack || map.isStyleLoaded()) return;
    fellBack = true;
    map.setStyle(plainStyle());
    el.append(h("div", { class: "note", style: { position: "absolute", left: "10px", right: "52px", bottom: "6px", zIndex: 3, background: "var(--panel)", padding: "4px 8px", borderRadius: "6px" } }, "Basemap unavailable: showing tracks without the map background."));
  };
  map.on("error", (e) => { if (!map.isStyleLoaded()) fallBack(); });
  setTimeout(fallBack, 15000);
  return map;
}

// tracks: [{flight_id, aircraft_label, coords:[[lon,lat],...], ...}]; colorBy: "route" | "aircraft"
export function addTracks(map, tracks, { colorBy = "aircraft", width = 1.6, opacity = 0.75, id = "tracks" } = {}) {
  const fc = {
    type: "FeatureCollection",
    features: tracks.map((t) => ({ type: "Feature", id: t.flight_id,
      properties: { flight_id: t.flight_id, aircraft: t.aircraft_label, color: colorBy === "route" ? css("--route") : acColor(t.aircraft_label),
                    start: t.start_utc, dep: t.dep, arr: t.arr, km: t.track_km, type: t.session_type },
      geometry: { type: "LineString", coordinates: t.coords } })),
  };
  if (map.getSource(id)) { map.getSource(id).setData(fc); return; }
  map.addSource(id, { type: "geojson", data: fc });
  map.addLayer({ id: id + "-glow", type: "line", source: id, layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": ["get", "color"], "line-width": width * 3.5, "line-opacity": opacity * 0.12, "line-blur": 2 } });
  map.addLayer({ id, type: "line", source: id, layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": ["get", "color"], "line-width": ["case", ["boolean", ["feature-state", "hover"], false], width * 2.4, width],
             "line-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 1, opacity] } });
  map.addLayer({ id: id + "-hl", type: "line", source: id, filter: ["==", ["get", "flight_id"], -1], layout: { "line-cap": "round" },
    paint: { "line-color": css("--route"), "line-width": width * 2.6, "line-opacity": 1 } });
}

export function highlight(map, fid, id = "tracks") {
  if (map && map.getLayer(id + "-hl")) map.setFilter(id + "-hl", ["==", ["get", "flight_id"], fid ?? -1]);
}

export function addAirports(map, airports, id = "airports") {
  // circles as a layer; codes as HTML markers (the raster style has no glyphs for symbol layers)
  const fc = { type: "FeatureCollection", features: airports.map((a) => ({ type: "Feature", properties: { code: a.code, name: a.name, fixes: a.fixes || 0 },
    geometry: { type: "Point", coordinates: [a.longitude_deg, a.latitude_deg] } })) };
  if (map.getSource(id)) map.getSource(id).setData(fc);
  else {
    map.addSource(id, { type: "geojson", data: fc });
    map.addLayer({ id, type: "circle", source: id, paint: { "circle-radius": 7, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": css("--accent"), "circle-stroke-width": 2 } });
  }
  (map._eplanesMarkers || []).forEach((m) => m.remove());
  map._eplanesMarkers = airports.map((a) => new window.maplibregl.Marker({
    element: h("div", { class: "display", style: { color: css("--ink"), fontSize: "12px", textShadow: `0 0 3px ${css("--bg")}, 0 0 3px ${css("--bg")}`, transform: "translateY(16px)", pointerEvents: "none" } }, a.code),
  }).setLngLat([a.longitude_deg, a.latitude_deg]).addTo(map));
}

export function fit(map, coordsList, padding = 40) {
  const b = bounds(coordsList);
  if (b) map.fitBounds(b, { padding, duration: 0, maxZoom: 12 });
}

export function hoverTracks(map, id, onHover) {
  let hovered = null;
  map.on("mousemove", id, (e) => {
    map.getCanvas().style.cursor = "pointer";
    const f = e.features[0];
    if (hovered !== null) map.setFeatureState({ source: id, id: hovered }, { hover: false });
    hovered = f.id;
    map.setFeatureState({ source: id, id: hovered }, { hover: true });
    if (onHover) onHover(f.properties);
  });
  map.on("mouseleave", id, () => {
    map.getCanvas().style.cursor = "";
    if (hovered !== null) map.setFeatureState({ source: id, id: hovered }, { hover: false });
    hovered = null;
    if (onHover) onHover(null);
  });
}

// Canvas fallback when WebGL is unavailable: equirectangular plot of the tracks, no basemap.
export function canvasFallback(el, tracks, airports = [], { colorBy = "aircraft", highlightId = null } = {}) {
  el.innerHTML = "";
  const box = bounds(tracks.map((t) => t.coords).concat(airports.map((a) => [[a.longitude_deg, a.latitude_deg]])));
  if (!box) { el.append(h("div", { class: "map-empty" }, "No GPS tracks to draw.")); return; }
  const c = h("canvas", { style: { width: "100%", height: "100%", display: "block" }, role: "img", "aria-label": "Flight tracks" });
  el.append(c, h("div", { class: "note", style: { position: "absolute", left: "10px", bottom: "6px" } }, "Simplified view: WebGL is unavailable, so the basemap is not shown."));
  const draw = () => {
    const r = el.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    c.width = r.width * dpr; c.height = r.height * dpr;
    const g = c.getContext("2d"); g.scale(dpr, dpr);
    g.fillStyle = css("--panel-2"); g.fillRect(0, 0, r.width, r.height);
    const [[w, s], [e, n]] = box, pad = 30, k = Math.cos(((s + n) / 2) * Math.PI / 180);
    const sx = (r.width - 2 * pad) / Math.max((e - w) * k, 1e-6), sy = (r.height - 2 * pad) / Math.max(n - s, 1e-6), sc = Math.min(sx, sy);
    const X = (x) => pad + (x - w) * k * sc + (r.width - 2 * pad - (e - w) * k * sc) / 2, Y = (y) => r.height - pad - (y - s) * sc - (r.height - 2 * pad - (n - s) * sc) / 2;
    for (const t of tracks) {
      g.strokeStyle = t.flight_id === highlightId ? css("--route") : colorBy === "route" ? css("--route") : acColor(t.aircraft_label);
      g.globalAlpha = t.flight_id === highlightId ? 1 : 0.6; g.lineWidth = t.flight_id === highlightId ? 3 : 1.3;
      g.beginPath(); t.coords.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)))); g.stroke();
    }
    g.globalAlpha = 1; g.strokeStyle = css("--accent"); g.lineWidth = 2; g.fillStyle = css("--ink"); g.font = "12px " + css("--font-display");
    for (const a of airports) { g.beginPath(); g.arc(X(a.longitude_deg), Y(a.latitude_deg), 6, 0, 2 * Math.PI); g.stroke(); g.fillText(a.code, X(a.longitude_deg) + 9, Y(a.latitude_deg) + 4); }
  };
  draw();
  new ResizeObserver(draw).observe(el);
}
