"""Pipistrel cloud portal access: login, aircraft discovery, flight download.

Behaviour is ported from the project notebooks (EDA_Data / Scrape_data_chrome):
  * flight IDs come from the paginated aircraft page (``tr.clickable-aircraft``
    rows whose onclick is ``details('../flight/', ID)``);
  * flight metadata comes from the ``table.white-link`` on the flight page;
  * the ZIP comes from the "Download CSV file" link;
  * 403 / timeouts are retried with exponential backoff, 404 is NOT retried
    (the portal has not finished processing that flight).

NOTE: this module was written without access to the live portal (the build
sandbox could not reach it). It is exercised by tests against mocked HTML only;
run ``eplanes discover`` first and read its report before a full download.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ONCLICK_RE = re.compile(r"details\(\s*'(?:\.\./)?flight/'\s*,\s*(\d+)\s*\)")
PAGE_RE = re.compile(r"Page\s+\d+\s+of\s+(\d+)")


class NotReady(Exception):
    """HTTP 404 on the CSV: not retried."""


def login_requests(base_url: str, login_path: str, user: str, pw: str) -> requests.Session:
    """Form login with requests. Raises if the session does not look logged in."""
    s = requests.Session()
    s.headers["User-Agent"] = "eplanes-dataset-pipeline/1.0 (research)"
    url = base_url + login_path
    r = s.get(url, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form")
    payload = {}
    if form:
        for inp in form.find_all("input"):
            if inp.get("name") and inp.get("type") in ("hidden",):
                payload[inp["name"]] = inp.get("value", "")
    payload.update({"username": user, "password": pw})
    action = urljoin(url, form.get("action") or url) if form else url
    r2 = s.post(action, data=payload, timeout=30, allow_redirects=True)
    if "login" in r2.url.lower() and 'name="password"' in r2.text:
        raise RuntimeError("requests-based login did not succeed")
    return s


def login_selenium(base_url: str, login_path: str, user: str, pw: str,
                   headless: bool = False) -> requests.Session:
    """Browser login (as in the original notebooks) then hand cookies to requests."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys

    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(base_url + login_path)
        time.sleep(2)
        driver.find_element(By.NAME, "username").send_keys(user)
        f = driver.find_element(By.NAME, "password")
        f.send_keys(pw)
        f.send_keys(Keys.RETURN)
        time.sleep(3)
        s = requests.Session()
        s.headers["User-Agent"] = "eplanes-dataset-pipeline/1.0 (research)"
        for ck in driver.get_cookies():
            s.cookies.set(ck["name"], ck["value"])
        return s
    finally:
        driver.quit()


def check_logged_in(session: requests.Session, base_url: str, aircraft_id: int) -> bool:
    r = session.get(f"{base_url}/electro/aircraft/{aircraft_id}", timeout=30)
    return r.status_code == 200 and 'name="password"' not in r.text


def login(base_url, login_path, user, pw, mode="auto", headless=False):
    if mode in ("auto", "requests"):
        try:
            return login_requests(base_url, login_path, user, pw)
        except Exception as e:  # noqa: BLE001
            if mode == "requests":
                raise
            print(f"[login] requests login failed ({e}); falling back to selenium")
    return login_selenium(base_url, login_path, user, pw, headless=headless)


# ── aircraft pages ────────────────────────────────────────────────────────────

def parse_flight_ids(html: str) -> tuple[list[int], int]:
    soup = BeautifulSoup(html, "html.parser")
    ids = []
    for row in soup.find_all("tr", class_="clickable-aircraft"):
        m = ONCLICK_RE.search(row.get("onclick", ""))
        if m:
            ids.append(int(m.group(1)))
    total = 1
    span = soup.find("span", class_=lambda c: c and "current" in c)
    if span:
        m = PAGE_RE.search(span.get_text())
        if m:
            total = int(m.group(1))
    return ids, total


def aircraft_page_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)[:2000]


def list_flights(session, base_url, aircraft_id, delay=0.3, log=print) -> list[int]:
    ids: set[int] = set()
    page, total = 1, None
    while True:
        r = session.get(f"{base_url}/electro/aircraft/{aircraft_id}?page={page}", timeout=30)
        if r.status_code != 200:
            log(f"  aircraft {aircraft_id} page {page}: HTTP {r.status_code}, stopping")
            break
        page_ids, tp = parse_flight_ids(r.text)
        if not page_ids:
            break
        ids.update(page_ids)
        total = total or tp
        log(f"  aircraft {aircraft_id} page {page}/{total}: {len(page_ids)} flights (total {len(ids)})")
        if page >= total:
            break
        page += 1
        time.sleep(delay)
    return sorted(ids)


# ── flight pages ──────────────────────────────────────────────────────────────

def parse_date(s: str) -> datetime | None:
    """Parse portal dates incl. 'midnight'/'noon'. Keeps time of day (the
    original notebooks kept the date only)."""
    if not s:
        return None
    t = s
    for word, hm in (("midnight", "12:00 am"), ("noon", "12:00 pm")):
        if word in t.lower():
            t = re.sub(word, hm, t, flags=re.I)
            break
    t = t.replace(".", "").strip()
    for fmt in ("%B %d, %Y, %I:%M %p", "%B %d, %Y, %I %p", "%b %d, %Y, %I:%M %p",
                "%b %d, %Y, %I %p", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            continue
    return None


@dataclass
class FlightPage:
    flight_id: int
    info: dict
    download_url: str | None
    raw_html_len: int


def parse_flight_page(fid: int, html: str, base_url: str) -> FlightPage:
    soup = BeautifulSoup(html, "html.parser")
    link = soup.find("a", string=lambda s: s and "Download CSV file" in s) or \
        soup.find("a", string=lambda s: s and "Download CSV" in s)
    url = None
    if link is not None and link.get("href"):
        url = urljoin(f"{base_url}/electro/flight/{fid}", link["href"])
    info = {}
    table = soup.find("table", class_="white-link")
    if table:
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) == 2:
                info[cols[0].get_text(strip=True)] = cols[1].get_text(strip=True)
    return FlightPage(fid, info, url, len(html))


def fetch_flight_page(session, base_url, fid, timeout=30) -> FlightPage | None:
    r = session.get(f"{base_url}/electro/flight/{fid}", timeout=timeout)
    if r.status_code != 200:
        return None
    return parse_flight_page(fid, r.text, base_url)


def download_zip(session, url, max_retries=3, backoff=2.0, timeout=60, log=print) -> bytes:
    last = None
    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                raise NotReady(url)
            r.raise_for_status()
            return r.content
        except NotReady:
            raise
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            last = e
        if attempt < max_retries:
            wait = backoff * 2 ** attempt
            log(f"    retry {attempt + 1}/{max_retries} in {wait:.0f}s ({last})")
            time.sleep(wait)
    raise last
