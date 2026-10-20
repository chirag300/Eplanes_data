"""Website bundle, GPS handling and API (synthetic data, no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from eplanes import geo


def test_nearest_airport_waterloo():
    code, name, d = geo.nearest_airport(43.4561, -80.3886, 3.0)
    assert code == "CYKF" and d < 2.0


def test_unsigned_longitude_is_resolved_by_airfield_matching():
    rng = np.random.default_rng(0)
    # logger writes magnitudes only: Waterloo appears as +80.38
    ep = pd.DataFrame({"aircraft_label": "X", "lat": 43.4608 + rng.normal(0, 0.002, 20), "lng": 80.3786 + rng.normal(0, 0.002, 20)})
    r = geo.resolve_signs(ep).iloc[0]
    assert r.resolved and r.lat_sign == 1 and r.lng_sign == -1


def test_unresolvable_positions_stay_unresolved():
    ep = pd.DataFrame({"aircraft_label": "X", "lat": [10.0] * 5, "lng": [10.0] * 5})  # open water, no airfield
    assert not geo.resolve_signs(ep).iloc[0].resolved


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from eplanes.build.clean import build_clean
    from eplanes.build.raw import build_raw
    from eplanes.config import load_settings
    from eplanes.quality.report import run_quality
    from eplanes.synthetic import make_synthetic
    from eplanes.web_export import build_release, export_web
    tmp = tmp_path_factory.mktemp("web")
    s = load_settings().with_dirs(tmp / "data", tmp / "outputs")
    make_synthetic(s.data_dir, 6)
    q = lambda *a, **k: None  # noqa: E731
    build_raw(s, log=q)
    build_clean(s, log=q)
    run_quality(s, log=q)
    out = export_web(s, make_zip=True, log=q)
    rel = build_release(s, "0.0.1-test", log=q)
    return s, out, rel


def test_bundle_contents(bundle):
    s, out, rel = bundle
    m = json.loads((out / "manifest.json").read_text())
    assert m["synthetic"] is True and m["n_sessions"] > 0
    ses = pd.read_parquet(out / "sessions.parquet")
    ser = pd.read_parquet(out / "series.parquet")
    assert set(ses.flight_id) >= set(ser.flight_id)
    # avionics-off channels and start-up zeros are missing, not zero
    assert not (ser["oat_c"] == 0).any()
    # GPS signs recovered: synthetic fleets sit in the western hemisphere
    assert (ser["lon"].dropna() < 0).all()
    flights = ses[(ses.session_type == "flight test") & ses.has_gps_track]
    assert flights.dep_airport.isin(["CYKF", "CYYJ"]).mean() > 0.9
    assert (rel / "release_manifest.json").exists() and (rel / "checksums.sha256").exists()
    man = json.loads((rel / "release_manifest.json").read_text())
    raw = [f for f in man["files"] if f["kind"] == "raw"]
    assert raw and all(f["bytes"] > 10_000 for f in raw), "raw ZIP archives should contain the portal exports"


def test_api(bundle):
    from fastapi.testclient import TestClient

    from eplanes.web.server import create_app
    s, out, rel = bundle
    c = TestClient(create_app(out, rel, settings=None))
    st = c.get("/api/status").json()
    assert st["mode"] == "live" and st["dataset"]["synthetic"] is True and st["portal_check_available"] is False
    ses = c.get("/api/sessions").json()
    fid = int(pd.read_parquet(out / "series.parquet", columns=["flight_id"]).flight_id.iloc[0])
    assert fid in [r[ses["columns"].index("flight_id")] for r in ses["rows"]]
    d = c.get(f"/api/sessions/{fid}").json()
    assert d["session"]["flight_id"] == fid
    assert c.get("/api/sessions/999999999").status_code == 404
    ser = c.get(f"/api/sessions/{fid}/series?max_points=100").json()
    assert len(ser["t_s"]) <= 100
    r = c.post("/api/refresh?portal=true").json()
    assert r["refresh"]["ok"] and r["portal"]["status"] == "unavailable"
    files = c.get("/api/downloads").json()["files"]
    ok = [f for f in files if f["available"]][0]
    assert c.get("/" + ok["url"]).status_code == 200
    assert c.get("/downloads/../manifest.json").status_code == 404
    assert c.get("/downloads/not_in_release.csv").status_code == 404
    assert c.get("/").status_code == 200 and "Eplanes" in c.get("/").text


def test_static_snapshot(bundle, tmp_path):
    from eplanes.web.server import build_static
    s, out, rel = bundle
    site = build_static(out, tmp_path / "site", rel)
    st = json.loads((site / "data" / "status.json").read_text())
    assert st["mode"] == "static"
    assert "mode: 'static'" in (site / "config.js").read_text()
    assert len(list((site / "data" / "sessions").glob("*.json"))) == st["dataset"]["n_sessions"]


def test_static_snapshot_links_downloads_to_hosted_files(bundle, tmp_path):
    from eplanes.web.server import build_static
    s, out, rel = bundle
    site = build_static(out, tmp_path / "site2", rel, downloads_url="https://zenodo.org/records/123/files/")
    dl = json.loads((site / "data" / "downloads.json").read_text())
    assert dl["files"] and all(f["url"].startswith("https://zenodo.org/records/123/files/") and f["url"].endswith("?download=1")
                               for f in dl["files"])
    assert not (site / "downloads").exists()  # nothing copied into the site


def test_publish_site_pushes_snapshot_to_a_git_repo(bundle, tmp_path, monkeypatch):
    import subprocess
    from eplanes.web.publish import publish_site
    s, out, rel = bundle
    remote = tmp_path / "site.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@t"), ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@t")):
        monkeypatch.setenv(k, v)
    log = []
    assert publish_site(s, str(remote), log=log.append)
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"], capture_output=True, text=True).stdout.split()
    assert "render.yaml" in files and "index.html" in files and "data/status.json" in files
