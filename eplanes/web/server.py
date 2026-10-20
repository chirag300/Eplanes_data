"""Eplanes website: a small FastAPI service over the web bundle, plus a static-snapshot builder.

Data flow
    pipeline -> `eplanes export-web` -> web bundle (parquet/json) -> this server -> browser
    The browser never sees credentials. Portal credentials stay in the server's environment / .env and are
    only used by the optional portal check (POST /api/refresh with portal=true).

What "Refresh data" does (live mode)
    1. re-reads the bundle from disk (a new `export-web` run is picked up without restarting), and
    2. optionally logs in to the Pipistrel portal server-side and lists every session per aircraft, reporting
       sessions that exist on the portal but are not yet in the processed dataset (with their portal metadata).
    Processing new sessions is a pipeline run (download -> all -> export-web); the site reports, it does not invent.

Static mode (`eplanes web-static`) writes the same API responses as JSON files so the site can be hosted
without a server (e.g. GitHub Pages). There, "Refresh data" re-fetches the published snapshot.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import shutil
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

STATIC_DIR = Path(__file__).parent / "static"

SESSION_COLS = [
    "flight_id", "aircraft_label", "fleet", "session_type", "activity_from_data", "start_utc", "end_utc", "duration_min",
    "n_rows", "has_gps_track", "dep_airport", "arr_airport", "track_km", "airborne_min", "alt_gain_m", "oat_c_median",
    "b1_soc_start", "b1_soc_end", "b2_soc_start", "b2_soc_end", "b1_soh_median", "b2_soh_median", "b1_soh_start", "b1_soh_end",
    "b2_soh_start", "b2_soh_end", "b1_ah_discharged", "b2_ah_discharged", "b1_ah_charged", "b2_ah_charged",
    "b1_energy_discharged_kwh", "b2_energy_discharged_kwh", "b1_energy_charged_kwh", "b2_energy_charged_kwh",
    "b1_i_peak_discharge_a", "b2_i_peak_discharge_a", "b1_power_peak_kw", "b2_power_peak_kw", "b1_temp_start_c", "b1_temp_max_c",
    "b2_temp_max_c", "b1_temp_delta_c", "b1_soh_cell_cells_last_min", "b1_soh_cell_cells_last_mean", "b2_soh_cell_cells_last_min",
    "b2_soh_cell_cells_last_mean", "b1_uniform_cells", "b2_uniform_cells", "frac_avionics_valid", "has_gap", "n_gaps", "max_gap_min",
    "scale_anomaly", "is_duplicate_content", "label_content_mismatch", "usable_B_profiles", "usable_C_soc_dod",
    "usable_D_charging", "usable_E_thermal", "usable_F_soh", "usable_G_capacity", "usable_exclusion_reason", "download_status",
]
SERIES_COLS = ["b1_current_a", "b2_current_a", "b1_voltage_v", "b2_voltage_v", "b1_soc_pct", "b2_soc_pct", "b1_soh_pct", "b2_soh_pct",
               "b1_min_cell_mv", "b1_max_cell_mv", "b2_min_cell_mv", "b2_max_cell_mv", "b1_max_cell_temp_c", "b2_max_cell_temp_c",
               "b1_avg_cell_temp_c", "b2_avg_cell_temp_c", "motor_power_kw", "pressure_alt_m", "ias", "oat_c", "ground_speed",
               "lat", "lon"]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _clean(v):
    """JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else round(float(v), 4)
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return None if pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%dT%H:%M:%SZ")
    if v is pd.NaT:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def records(df: pd.DataFrame) -> list[dict]:
    cols = list(df.columns)
    return [{c: _clean(v) for c, v in zip(cols, row)} for row in df.itertuples(index=False, name=None)]


def decimate(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


class Store:
    """Everything the site serves, loaded from one bundle directory."""

    def __init__(self, bundle: Path, release: Path | None = None):
        self.bundle = Path(bundle)
        self.release = Path(release) if release else None
        self.lock = threading.RLock()
        self.loaded_at = None
        self.load()

    # ---- loading --------------------------------------------------------------------------------------------
    def load(self) -> dict:
        b = self.bundle
        need = ["manifest.json", "sessions.parquet", "series.parquet"]
        missing = [f for f in need if not (b / f).exists()]
        if missing:
            raise FileNotFoundError(f"web bundle {b} is missing {missing}; run `python -m eplanes export-web` first")
        manifest = json.loads((b / "manifest.json").read_text())
        sessions = pd.read_parquet(b / "sessions.parquet")
        for c in ("start_utc", "end_utc"):
            if c in sessions:
                sessions[c] = pd.to_datetime(sessions[c], errors="coerce")
        packs = pd.read_parquet(b / "packs.parquet") if (b / "packs.parquet").exists() else pd.DataFrame()
        steps = pd.read_parquet(b / "soh_steps.parquet") if (b / "soh_steps.parquet").exists() else pd.DataFrame()
        aircraft = json.loads((b / "aircraft.json").read_text()) if (b / "aircraft.json").exists() else []
        airports = pd.read_csv(b / "airports.csv") if (b / "airports.csv").exists() else pd.DataFrame()
        quality = json.loads((b / "quality.json").read_text()) if (b / "quality.json").exists() else {}
        dictionary = pd.read_csv(b / "data_dictionary.csv") if (b / "data_dictionary.csv").exists() else pd.DataFrame()
        rel_manifest = None
        for cand in ([self.release / "release_manifest.json"] if self.release else []) + [b / "release_manifest.json"]:
            if cand.exists():
                rel_manifest = json.loads(cand.read_text())
                break
        qtables = {p.name.removeprefix("quality__").removesuffix(".csv"): pd.read_csv(p) for p in sorted(b.glob("quality__*.csv"))}
        series = pd.read_parquet(b / "series.parquet")
        series = series.sort_values(["flight_id", "t_s"]).reset_index(drop=True)
        fid = series["flight_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, fid[1:] != fid[:-1]])
        index = {int(fid[s]): (int(s), int(e)) for s, e in zip(starts, np.r_[starts[1:], len(fid)])}
        with self.lock:
            prev = getattr(self, "manifest", None)
            self.manifest, self.sessions, self.packs, self.steps = manifest, sessions, packs, steps
            self.aircraft, self.airports, self.quality, self.dictionary = aircraft, airports, quality, dictionary
            self.release_manifest, self.qtables = rel_manifest, qtables
            self.series, self.series_index = series, index
            self.loaded_at = _now()
            self._tracks_cache = None
        changed = prev is None or prev.get("created_utc") != manifest.get("created_utc")
        return dict(changed=changed, created_utc=manifest.get("created_utc"), n_sessions=len(sessions))

    # ---- queries --------------------------------------------------------------------------------------------
    def status(self) -> dict:
        m = self.manifest
        s = self.sessions
        latest = s.sort_values("start_utc").dropna(subset=["start_utc"]).tail(1)
        return dict(dataset=dict(created_utc=m.get("created_utc"), pipeline_commit=m.get("pipeline_commit"), synthetic=m.get("synthetic", False),
                                 cleaning_rule=m.get("cleaning_rule"), resolution_s=m.get("resolution_s"), n_sessions=int(len(s)),
                                 n_aircraft=int(s.aircraft_label.nunique()), first_session_utc=_clean(s.start_utc.min()),
                                 last_session_utc=_clean(s.start_utc.max())),
                    latest_session=records(latest[["flight_id", "aircraft_label", "session_type", "start_utc", "duration_min"]])[0]
                    if len(latest) else None,
                    loaded_at=self.loaded_at)

    def sessions_payload(self) -> dict:
        cols = [c for c in SESSION_COLS if c in self.sessions]
        df = self.sessions[cols].sort_values("start_utc", ascending=False, na_position="last")
        return dict(columns=cols, rows=[[_clean(v) for v in r] for r in df.itertuples(index=False, name=None)])

    def session_detail(self, fid: int) -> dict | None:
        s = self.sessions[self.sessions.flight_id == fid]
        if s.empty:
            return None
        row = records(s)[0]
        packs = records(self.packs[self.packs.flight_id == fid]) if len(self.packs) else []
        steps = records(self.steps[self.steps.flight_id == fid]) if len(self.steps) else []
        same = self.sessions[self.sessions.aircraft_label == row["aircraft_label"]].sort_values("start_utc").reset_index(drop=True)
        i = int(same.index[same.flight_id == fid][0])
        nb = dict(previous=int(same.flight_id[i - 1]) if i > 0 else None, next=int(same.flight_id[i + 1]) if i + 1 < len(same) else None)
        return dict(session=row, packs=packs, soh_steps=steps, neighbours=nb, notes=self.session_notes(row),
                    has_series=fid in self.series_index)

    @staticmethod
    def session_notes(r: dict) -> list[dict]:
        n = []
        if r.get("scale_anomaly"):
            n.append(dict(level="serious", text="Current and voltage in this session are mis-scaled (logging fault). Electrical values are not physical; SOC and SOH are unaffected."))
        if r.get("is_duplicate_content"):
            n.append(dict(level="warning", text="Byte-identical copy of an earlier session; excluded from statistics."))
        if r.get("label_content_mismatch"):
            n.append(dict(level="warning", text=f"The portal labels this session '{r.get('session_type')}', but the measured current says '{r.get('activity_from_data')}'."))
        if r.get("has_gap"):
            ng, gm = r.get("n_gaps"), float(r.get("max_gap_min") or 0)
            count = f"{int(ng)} logging gap{'s' if int(ng) != 1 else ''}" if ng else "Logging gaps"
            longest = f"{gm * 60:.0f} s" if gm < 1 else f"{gm:.1f} min"
            n.append(dict(level="info", text=f"{count}, the longest {longest}; charge is not integrated across gaps."))
        if (r.get("frac_avionics_valid") or 0) < 0.05:
            n.append(dict(level="info", text="Avionics were off: no outside-air temperature, altitude or GPS for this session."))
        if r.get("b1_uniform_cells") or r.get("b2_uniform_cells"):
            n.append(dict(level="warning", text="All 96 cells report the same SOH and capacity, the signature of a BMS estimator reset."))
        return n

    def series_payload(self, fid: int, max_points: int = 2000, cols: list[str] | None = None) -> dict | None:
        if fid not in self.series_index:
            return None
        a, b = self.series_index[fid]
        g = self.series.iloc[a:b]
        idx = decimate(len(g), max_points)
        g = g.iloc[idx]
        cols = [c for c in (cols or SERIES_COLS) if c in g]
        out = dict(flight_id=fid, n_bins=int(b - a), resolution_s=self.manifest.get("resolution_s"), t_s=[_clean(v) for v in g.t_s],
                   start_utc=_clean(g.utc.iloc[0]) if "utc" in g and len(g) else None, columns={})
        for c in cols:
            v = g[c].to_numpy(dtype=float)
            if np.isfinite(v).any():
                out["columns"][c] = [None if not np.isfinite(x) else round(float(x), 6 if c in ("lat", "lon") else 3) for x in v]
        return out

    def tracks(self, max_points: int = 160) -> list[dict]:
        if self._tracks_cache is not None:
            return self._tracks_cache
        out = []
        if "lat" not in self.series:
            self._tracks_cache = out
            return out
        meta = self.sessions.set_index("flight_id")
        for fid, (a, b) in self.series_index.items():
            g = self.series.iloc[a:b][["lat", "lon", "pressure_alt_m"]].dropna(subset=["lat", "lon"])
            if len(g) < 10:
                continue
            g = g.iloc[decimate(len(g), max_points)]
            m = meta.loc[fid]
            out.append(dict(flight_id=int(fid), aircraft_label=m.aircraft_label, session_type=m.session_type,
                            start_utc=_clean(m.start_utc), dep=_clean(m.get("dep_airport")), arr=_clean(m.get("arr_airport")),
                            track_km=_clean(m.get("track_km")),
                            coords=[[round(float(lo), 5), round(float(la), 5)] for la, lo in zip(g.lat, g.lon)]))
        self._tracks_cache = out
        return out

    def airports_payload(self) -> list[dict]:
        if self.airports.empty:
            return []
        s = self.sessions
        cnt = pd.concat([s.dep_airport, s.arr_airport]).value_counts()
        a = self.airports.copy()
        a["fixes"] = a.code.map(cnt).fillna(0).astype(int)
        return records(a)

    def aircraft_payload(self) -> list[dict]:
        s = self.sessions
        out = []
        for a in self.aircraft:
            g = s[s.aircraft_label == a["label"]]
            ft = g[g.session_type.astype(str).str.contains("flight", case=False, na=False)]
            soh_last = g.sort_values("start_utc").dropna(subset=["b1_soh_median"]).tail(1)
            out.append(dict(a, n_sessions=int(len(g)), n_flight_tests=int(len(ft)),
                            n_charging=int((g.session_type == "charging").sum()), hours_logged=_clean(g.duration_min.sum() / 60),
                            airborne_hours=_clean(g.airborne_min.sum() / 60) if "airborne_min" in g else None,
                            latest_soh_b1=_clean(soh_last.b1_soh_median.iat[0]) if len(soh_last) else None,
                            latest_soh_b2=_clean(soh_last.b2_soh_median.iat[0]) if len(soh_last) else None,
                            first_utc=_clean(g.start_utc.min()), last_utc=_clean(g.start_utc.max())))
        return out

    def quality_payload(self) -> dict:
        return dict(gate=self.quality, tables={k: records(v) for k, v in self.qtables.items()})

    def downloads_payload(self) -> dict:
        m = self.release_manifest
        if not m:
            return dict(available=False, message="No dataset release has been built yet. Run `python -m eplanes release`.")
        files = []
        for f in m.get("files", []):
            local = bool(self.release and (self.release / f["file"]).exists())
            files.append(dict(f, available=local, url=f"downloads/{f['file']}" if local else None))
        return dict(m, files=files, available=True)


class PortalChecker:
    """Lists sessions on the Pipistrel portal (server side) and compares them with the processed dataset."""

    MIN_INTERVAL_S = 300

    def __init__(self, settings, store: Store):
        self.settings = settings
        self.store = store
        self.state = dict(status="idle", checked_at=None, message="The portal has not been checked in this server run.", aircraft=[])
        self.lock = threading.Lock()
        self._last_start = 0.0
        cache = store.bundle.parent / "portal_check.json"
        self.cache = cache
        if cache.exists():
            try:
                self.state = json.loads(cache.read_text())
                self.state["status"] = "idle"
            except Exception:  # noqa: BLE001
                pass

    def available(self) -> tuple[bool, str]:
        if self.settings is None:
            return False, "No pipeline configuration loaded."
        from ..config import credentials
        try:
            for acct in self.settings["accounts"]:
                credentials(acct, self.settings)
        except Exception as e:  # noqa: BLE001
            return False, f"Portal credentials are not configured on this server ({str(e).split(':')[0]})."
        return True, "ok"

    def start(self) -> dict:
        with self.lock:
            if self.state.get("status") == "running":
                return self.state
            ok, why = self.available()
            if not ok:
                self.state.update(status="unavailable", message=why)
                return self.state
            if time.time() - self._last_start < self.MIN_INTERVAL_S:
                self.state.update(message=f"Checked {int(time.time() - self._last_start)} s ago; the portal is checked at most every "
                                          f"{self.MIN_INTERVAL_S // 60} minutes.")
                return self.state
            self._last_start = time.time()
            self.state.update(status="running", message="Logging in to the Pipistrel portal and listing sessions.", started_at=_now())
        threading.Thread(target=self._run, daemon=True).start()
        return self.state

    def _run(self):
        from ..acquisition import portal
        from ..config import credentials
        cfg = self.settings["portal"]
        known = set(int(x) for x in self.store.sessions.flight_id)
        rows = []
        try:
            for acct, a in self.settings["accounts"].items():
                user, pw = credentials(acct, self.settings)
                sess = portal.login(cfg["base_url"], cfg["login_path"], user, pw, "requests", verify_aircraft_id=a["aircraft"][0]["aircraft_id"])
                for ac in a["aircraft"]:
                    ids = portal.list_flights(sess, cfg["base_url"], ac["aircraft_id"], 0.2, log=lambda *x: None)
                    new = sorted(set(ids) - known)
                    meta = []
                    for fid in new[-15:]:
                        p = portal.fetch_flight_page(sess, cfg["base_url"], fid)
                        if p is not None:
                            when = portal.parse_date(p.info.get("Date", ""))
                            meta.append(dict(flight_id=fid, date=_clean(pd.Timestamp(when)) if when else p.info.get("Date"),
                                             session_type=p.info.get("Flight type"), duration=p.info.get("Duration"),
                                             has_csv=bool(p.download_url)))
                    rows.append(dict(aircraft_label=ac["label"], on_portal=len(ids), in_dataset=len(set(ids) & known),
                                     not_yet_processed=len(new), newest_on_portal=max(ids) if ids else None, new_sessions=meta))
            n_new = sum(r["not_yet_processed"] for r in rows)
            msg = ("The processed dataset is up to date with the portal." if n_new == 0 else
                   f"{n_new} session(s) are on the portal but not yet in the processed dataset. "
                   "Run the pipeline (download, all, export-web) to add them.")
            self.state = dict(status="done", checked_at=_now(), message=msg, aircraft=rows)
        except Exception as e:  # noqa: BLE001
            self.state = dict(status="error", checked_at=_now(), aircraft=rows,
                              message=f"Portal check failed: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
        try:
            self.cache.write_text(json.dumps(self.state, indent=1, default=str))
        except Exception:  # noqa: BLE001
            pass


def create_app(bundle: Path, release: Path | None = None, settings=None):
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    store = Store(bundle, release)
    checker = PortalChecker(settings, store)
    refresh_state = dict(at=store.loaded_at, ok=True, message="Loaded at server start.")
    app = FastAPI(title="Eplanes", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    def nocache(payload, status=200):
        return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        return nocache(dict(mode="live", **store.status(), refresh=refresh_state, portal=checker.state,
                            portal_check_available=checker.available()[0]))

    @app.post("/api/refresh")
    def refresh(portal: bool = False):
        try:
            res = store.load()
            refresh_state.update(at=_now(), ok=True,
                                 message=("New dataset version loaded." if res["changed"] else "Dataset re-read from disk; no new export since the last load."))
        except Exception as e:  # noqa: BLE001
            refresh_state.update(at=_now(), ok=False, message=f"Could not reload the dataset: {e}")
            return nocache(dict(refresh=refresh_state, portal=checker.state), 500)
        if portal:
            checker.start()
        return nocache(dict(refresh=refresh_state, portal=checker.state, **store.status()))

    @app.get("/api/sessions")
    def sessions():
        return store.sessions_payload()

    @app.get("/api/sessions/{fid}")
    def session(fid: int):
        d = store.session_detail(fid)
        if d is None:
            raise HTTPException(404, f"Session {fid} is not in the dataset.")
        return d

    @app.get("/api/sessions/{fid}/series")
    def series(fid: int, max_points: int = Query(2000, ge=50, le=20000), cols: str | None = None):
        d = store.series_payload(fid, max_points, cols.split(",") if cols else None)
        if d is None:
            raise HTTPException(404, f"No telemetry for session {fid}.")
        return d

    @app.get("/api/sessions/{fid}/trace.csv")
    def trace_csv(fid: int):
        if fid not in store.series_index:
            raise HTTPException(404, f"No telemetry for session {fid}.")
        a, b = store.series_index[fid]
        buf = io.StringIO()
        store.series.iloc[a:b].to_csv(buf, index=False)
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="eplanes_session_{fid}_2s.csv"'})

    @app.get("/api/sessions.csv")
    def sessions_csv(ids: str = ""):
        df = store.sessions
        if ids:
            keep = {int(x) for x in ids.split(",") if x.strip().isdigit()}
            df = df[df.flight_id.isin(keep)]
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="eplanes_sessions_selection.csv"'})

    @app.get("/api/tracks")
    def tracks():
        return store.tracks()

    @app.get("/api/airports")
    def airports():
        return store.airports_payload()

    @app.get("/api/aircraft")
    def aircraft():
        return store.aircraft_payload()

    @app.get("/api/quality")
    def quality():
        return store.quality_payload()

    @app.get("/api/dictionary")
    def dictionary():
        return records(store.dictionary)

    @app.get("/api/downloads")
    def downloads():
        return store.downloads_payload()

    @app.get("/downloads/{name}")
    def download(name: str):
        m = store.release_manifest or {}
        allowed = {f["file"] for f in m.get("files", [])}
        if name not in allowed or not store.release or not (store.release / name).exists():
            raise HTTPException(404, "File not in the current release.")
        return FileResponse(store.release / name, filename=name)

    @app.get("/config.js")
    def config_js():
        return Response("window.EPLANES_CONFIG = {mode: 'live', api: 'api/'};", media_type="application/javascript",
                        headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def serve(bundle: Path, release: Path | None = None, host: str = "127.0.0.1", port: int = 8000, settings=None):
    import uvicorn
    if release is None and settings is not None:
        rels = sorted((settings.outputs_dir / "release").glob("v*")) if (settings.outputs_dir / "release").exists() else []
        release = rels[-1] if rels else None
    app = create_app(bundle, release, settings)
    print(f"Eplanes website on http://{host}:{port}  (bundle: {bundle}; release: {release or 'none'})")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def build_static(bundle: Path, out: Path, release: Path | None = None, copy_release: bool = False,
                 downloads_url: str | None = None) -> Path:
    """Write the site plus every API response as JSON files; no server needed.

    Release files are linked, in order of preference, to ``downloads_url`` (where they are hosted, e.g. a Zenodo
    record's files), to copies inside the site (``copy_release``), or listed without a link."""
    store = Store(bundle, release)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC_DIR, out)
    d = out / "data"
    (d / "sessions").mkdir(parents=True)
    (d / "series").mkdir()

    def dump(path: Path, obj):
        path.write_text(json.dumps(obj, separators=(",", ":"), default=_clean))

    st = store.status()
    dump(d / "status.json", dict(mode="static", **st, published_at=_now(),
                                 refresh=dict(at=_now(), ok=True, message="Published snapshot."), portal=None, portal_check_available=False))
    dump(d / "sessions.json", store.sessions_payload())
    for fid in store.sessions.flight_id:
        dump(d / "sessions" / f"{int(fid)}.json", store.session_detail(int(fid)))
        s = store.series_payload(int(fid), 2000)
        if s:
            dump(d / "series" / f"{int(fid)}.json", s)
    dump(d / "tracks.json", store.tracks())
    dump(d / "airports.json", store.airports_payload())
    dump(d / "aircraft.json", store.aircraft_payload())
    dump(d / "quality.json", store.quality_payload())
    dump(d / "dictionary.json", records(store.dictionary))
    dl = store.downloads_payload()
    if dl.get("available") and downloads_url:
        from urllib.parse import quote
        base = downloads_url.rstrip("/")
        for f in dl["files"]:
            f.update(available=True, url=f"{base}/{quote(f['file'])}" + ("?download=1" if "zenodo.org" in base else ""))
    elif dl.get("available") and copy_release and release:
        (out / "downloads").mkdir()
        for f in dl["files"]:
            if (release / f["file"]).exists():
                shutil.copy(release / f["file"], out / "downloads" / f["file"])
                f.update(available=True, url=f"downloads/{f['file']}")
    dump(d / "downloads.json", dl)
    (out / "config.js").write_text("window.EPLANES_CONFIG = {mode: 'static', api: 'data/'};\n")
    print(f"static site written to {out}")
    return out
