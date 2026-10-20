import * as api from "../api.js";
import { fmt, h } from "../ui.js";

export async function renderAbout(app) {
  const st = await api.status();
  app.innerHTML = "";
  const wrap = h("div", { class: "wrap" });
  app.append(wrap);
  wrap.append(h("div", { class: "section prose" },
    h("h1", { style: { marginBottom: "16px" } }, "About this project"),
    h("p", { style: { fontSize: "18px" } }, "Laboratory ageing tests cycle single cells under controlled profiles and measure capacity directly. Aircraft in service do neither: they fly partial cycles in changing weather and report health through their battery management system. This site publishes what such an aircraft actually records, so that battery-health methods can be tested on operational data."),
    h("h3", {}, "Where the data come from"),
    h("p", {}, "The Pipistrel Velis Electro uploads a log of every powered session to the manufacturer's cloud portal. The Eplanes pipeline downloads each session's export from the operators' portal accounts, keeps the originals unchanged, and builds the cleaned tables, traces and quality flags served here."),
    h("h3", {}, "What the site shows"),
    h("p", {}, "Historical records only. Sessions appear after the pipeline has processed them; nothing here is live tracking of aircraft. The status line in the header gives the time of the last refresh and the newest session in the dataset."),
    h("h3", {}, "Reading battery health"),
    h("p", {}, "The state of health on these pages is the BMS's estimate for its weakest cell. It is revised mostly while charging and can step up as well as down, so it should be read as an estimator output rather than a measured capacity. The dataset paper discusses which health indicators can and cannot be recovered from these signals."),
    h("h3", {}, "Data status"),
    h("p", {}, `Current export: ${fmt.dateTime(st.dataset.created_utc)}, ${fmt.int(st.dataset.n_sessions)} sessions, newest ${fmt.date(st.dataset.last_session_utc)}.`),
    h("h3", {}, "Affiliations"),
    h("p", {}, "University of Waterloo, Waterloo Institute for Sustainable Aeronautics (WISA), and the Region of Waterloo International Airport. Logos are shown in the footer when the official files are installed; their presence does not imply endorsement of any analysis."),
    h("h3", {}, "Contact"),
    h("p", {}, "Questions about the dataset: Chirag Seth, University of Waterloo.")));
}
