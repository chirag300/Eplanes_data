"""Resumable bulk download with a manifest.

Layout written (all under data/raw/):
    zips/<aircraft_label>/flight_<id>.zip      original ZIP, byte-for-byte
    meta/<aircraft_label>/flight_<id>.json     everything parsed from the flight page
    manifest.csv                               one row per flight: status, sha256, ...
Re-running skips flights whose ZIP is already valid; ``--retry-failed`` retries errors.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd

from ..config import Settings, credentials
from . import portal

MANIFEST_COLS = ["flight_id", "account", "aircraft_id", "aircraft_label", "status", "http_note",
                 "zip_path", "meta_path", "sha256", "zip_bytes", "n_members", "downloaded_utc"]


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _load_manifest(p: Path) -> pd.DataFrame:
    if p.exists():
        return pd.read_csv(p, dtype={"flight_id": int})
    return pd.DataFrame(columns=MANIFEST_COLS)


def _valid_zip(p: Path) -> bool:
    try:
        with zipfile.ZipFile(p) as z:
            return z.testzip() is None and len(z.namelist()) > 0
    except Exception:  # noqa: BLE001
        return False


def run_download(settings: Settings, accounts: list[str] | None = None,
                 retry_failed: bool = False, limit: int | None = None, log=print) -> pd.DataFrame:
    cfg = settings["portal"]
    dl = settings["download"]
    raw_dir = settings.data_dir / "raw"
    man_path = raw_dir / "manifest.csv"
    manifest = _load_manifest(man_path)
    done = {int(r.flight_id): r for r in manifest.itertuples()}
    rows = {int(r.flight_id): r._asdict() for r in manifest.itertuples()}
    n_new = 0

    for acct in accounts or list(settings["accounts"]):
        user, pw = credentials(acct, settings)
        log(f"[{acct}] logging in")
        first = settings["accounts"][acct]["aircraft"][0]["aircraft_id"]
        sess = portal.login(cfg["base_url"], cfg["login_path"], user, pw, dl["login_mode"], verify_aircraft_id=first)
        for ac in settings["accounts"][acct]["aircraft"]:
            aid, label = ac["aircraft_id"], ac["label"]
            if not portal.check_logged_in(sess, cfg["base_url"], aid):
                log(f"[{acct}] aircraft {aid} ({label}): page lists no flights for this login, skipping")
                continue
            log(f"[{acct}] listing flights for {label} (aircraft {aid})")
            fids = portal.list_flights(sess, cfg["base_url"], aid, dl["delay_s"], log)
            for k, fid in enumerate(fids, 1):
                prev = done.get(fid)
                if prev is not None:
                    ok = prev.status == "ok" and Path(prev.zip_path).exists() and _valid_zip(Path(prev.zip_path))
                    if ok or (prev.status in ("not_ready", "no_csv") and not retry_failed):
                        continue
                    if prev.status == "error" and not retry_failed:
                        continue
                if limit is not None and n_new >= limit:
                    break
                log(f"[{label} {k}/{len(fids)}] flight {fid}")
                rec = dict(flight_id=fid, account=acct, aircraft_id=aid, aircraft_label=label,
                           status="", http_note="", zip_path="", meta_path="", sha256="",
                           zip_bytes=0, n_members=0)
                try:
                    page = portal.fetch_flight_page(sess, cfg["base_url"], fid, dl["timeout_s"])
                    if page is None:
                        rec.update(status="error", http_note="flight page not 200")
                    else:
                        meta = {"flight_id": fid, "account": acct, "aircraft_id": aid,
                                "aircraft_label": label, "page_info": page.info,
                                "download_url": page.download_url}
                        mp = raw_dir / "meta" / label / f"flight_{fid}.json"
                        _atomic_write(mp, json.dumps(meta, indent=1).encode())
                        rec["meta_path"] = str(mp)
                        if not page.download_url:
                            rec.update(status="no_csv", http_note="no download link")
                        else:
                            data = portal.download_zip(sess, page.download_url, dl["max_retries"],
                                                       dl["backoff_base_s"], dl["timeout_s"], log)
                            with zipfile.ZipFile(io.BytesIO(data)) as z:  # raises BadZipFile
                                names = z.namelist()
                            zp = raw_dir / "zips" / label / f"flight_{fid}.zip"
                            _atomic_write(zp, data)
                            rec.update(status="ok", zip_path=str(zp),
                                       sha256=hashlib.sha256(data).hexdigest(),
                                       zip_bytes=len(data), n_members=len(names))
                except portal.NotReady:
                    rec.update(status="not_ready", http_note="404 on CSV (not retried)")
                except zipfile.BadZipFile:
                    rec.update(status="error", http_note="BadZipFile")
                except Exception as e:  # noqa: BLE001
                    rec.update(status="error", http_note=f"{type(e).__name__}: {e}"[:200])
                rec["downloaded_utc"] = pd.Timestamp.utcnow().isoformat()
                rows[fid] = rec
                n_new += 1
                if n_new % 10 == 0:  # checkpoint
                    pd.DataFrame(rows.values())[MANIFEST_COLS].to_csv(man_path, index=False)
                time.sleep(dl["delay_s"])
    man_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows.values())
    out = out.reindex(columns=MANIFEST_COLS)
    out.to_csv(man_path, index=False)
    log(out.groupby(["aircraft_label", "status"]).size().to_string())
    return out


def run_discover(settings: Settings, accounts: list[str] | None = None, log=print) -> pd.DataFrame:
    """Verify aircraft IDs / counts before a full download."""
    cfg = settings["portal"]
    rows = []
    for acct in accounts or list(settings["accounts"]):
        user, pw = credentials(acct, settings)
        first = settings["accounts"][acct]["aircraft"][0]["aircraft_id"]
        try:
            sess = portal.login(cfg["base_url"], cfg["login_path"], user, pw, settings["download"]["login_mode"],
                                verify_aircraft_id=first)
        except Exception as e:  # noqa: BLE001
            log(f"[{acct}] LOGIN FAILED: {e}")
            rows += [dict(account=acct, aircraft_id=a["aircraft_id"], expected_label=a["label"], http=None,
                          logged_in=False, label_found_on_page=False, n_flights=0, min_id=None, max_id=None)
                     for a in settings["accounts"][acct]["aircraft"]]
            continue
        for ac in settings["accounts"][acct]["aircraft"]:
            r = sess.get(f"{cfg['base_url']}/electro/aircraft/{ac['aircraft_id']}", timeout=30)
            dbg = settings.data_dir / "raw" / "debug" / f"aircraft_{ac['aircraft_id']}.html"
            dbg.parent.mkdir(parents=True, exist_ok=True)
            dbg.write_text(r.text)  # git-ignored; inspect if parsing finds no flights
            fids = portal.list_flights(sess, cfg["base_url"], ac["aircraft_id"], 0.3, log) if r.status_code == 200 else []
            txt = portal.aircraft_page_text(r.text) if r.status_code == 200 else ""
            rows.append(dict(account=acct, aircraft_id=ac["aircraft_id"], expected_label=ac["label"],
                             http=r.status_code, logged_in=not portal.looks_like_login_page(r.text),
                             label_found_on_page=ac["label"] in txt,
                             n_flights=len(fids), min_id=min(fids, default=None), max_id=max(fids, default=None)))
    df = pd.DataFrame(rows)
    out = settings.data_dir / "raw" / "aircraft_inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log(df.to_string(index=False))
    return df
