// ECharts wrapper: themed from CSS tokens, resized with its container, disposed on route change.
import { css } from "./ui.js";

const live = new Set();

export function disposeAll() {
  for (const c of live) { try { c.dispose(); } catch (e) { /* already gone */ } }
  live.clear();
}

export function tokens() {
  return {
    ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"), line: css("--line"), lineStrong: css("--line-strong"),
    panel: css("--panel"), panel2: css("--panel-2"), accent: css("--accent"), route: css("--route"), caution: css("--caution"),
    good: css("--good"), serious: css("--serious"), critical: css("--critical"),
    s: [css("--s1"), css("--s2"), css("--s3"), css("--s4")], pack1: css("--pack1"), pack2: css("--pack2"),
    seq: [0, 1, 2, 3, 4, 5].map((i) => css(`--seq-${i}`)),
    font: css("--font-body"), display: css("--font-display"),
  };
}

export function axis(t, extra = {}) {
  return {
    axisLine: { lineStyle: { color: t.lineStrong } },
    axisTick: { show: false },
    axisLabel: { color: t.ink3, fontSize: 12, hideOverlap: true },
    splitLine: { lineStyle: { color: t.line } },
    nameTextStyle: { color: t.ink2, fontSize: 12 },
    ...extra,
  };
}

export function tooltip(t, extra = {}) {
  return {
    backgroundColor: t.panel, borderColor: t.lineStrong, borderWidth: 1, padding: [8, 10],
    textStyle: { color: t.ink, fontSize: 13, fontFamily: t.font },
    extraCssText: "box-shadow: 0 6px 18px rgba(0,0,0,.25); border-radius: 6px;",
    ...extra,
  };
}

export function legend(t, extra = {}) {
  return { top: 0, left: 0, itemWidth: 14, itemHeight: 8, icon: "roundRect", textStyle: { color: t.ink2, fontSize: 12 }, ...extra };
}

export function make(el, buildOption) {
  if (!window.echarts) {
    el.innerHTML = '<div class="empty">The chart library did not load. Reload the page.</div>';
    return null;
  }
  const c = window.echarts.init(el, null, { renderer: "canvas" });
  live.add(c);
  const t = tokens();
  const opt = buildOption(t);
  c.setOption({ textStyle: { fontFamily: t.font, color: t.ink2 }, animationDuration: 400, ...opt });
  const ro = new ResizeObserver(() => c.resize());
  ro.observe(el);
  c.on("finished", () => {});
  return c;
}

// Common "no data" state rendered inside a chart container.
export function emptyChart(el, text) {
  el.innerHTML = "";
  el.style.display = "grid";
  el.style.placeItems = "center";
  el.append(Object.assign(document.createElement("div"), { className: "faint", textContent: text, style: "font-size:14px;text-align:center;padding:16px" }));
}
