# Eplanes website: running, publishing and hosting

The website is the public access point for the dataset. The code repository holds the code that builds it; it holds no data.
Nothing in this guide has been deployed. The site stays private until the paper is published and the data owners
have approved the release.

## 1. Two ways to run the site

| | Live mode (`eplanes web`) | Static snapshot (`eplanes web-static`) |
|---|---|---|
| What it is | A FastAPI server that reads the web bundle and serves the site and API | Plain files: the site plus every API response written as JSON |
| "Refresh data" | Re-reads the bundle from disk. An optional portal check logs in **on the server**, using your `.env`, and lists sessions not yet processed | Re-fetches the published files; says when the snapshot was published |
| Needs | Python and the bundle on the machine that serves it | Any static web host |
| Credentials | Read from `.env` on the server; never sent to the browser | None. The snapshot contains no credentials |
| Use it for | Your own machine, now | The public site after publication |

New sessions reach the site through the pipeline, not through the website: `download` -> `all` -> `export-web` -> (`release`) -> restart or rebuild the site.

## 2. Run it privately on your machine (now)

```bash
git pull
python -m eplanes quality          # re-run QC (includes the cross-aircraft duplicate check)
python -m eplanes export-web       # outputs/web_bundle/
python -m eplanes release          # outputs/release/ (downloadable dataset + checksums)
python -m eplanes web --release-dir outputs/release
# open http://127.0.0.1:8000
```

The server binds to `127.0.0.1` by default, so only your computer can reach it. Use `--host 0.0.0.0` only on a network you trust.

## 3. Build the static snapshot

```bash
python -m eplanes web-static --out outputs/web_static \
    --downloads-url https://zenodo.org/records/<record-id>/files
```

- `--downloads-url` links every file on the Dataset page to where it is hosted, such as a Zenodo record. Zenodo links get `?download=1`.
- `--copy-release` copies the release files into the site instead. Use it only for small releases: most static hosts cap single files at 25-100 MB.
- With neither flag, the files are listed without links.
- Expect a few hundred MB and a few thousand files; most of it is one 2 s trace file per session.

Before uploading the snapshot anywhere, check that none of the values in your `.env` appear in it. The check prints file names only, never the values:

```bash
python - <<'EOF'
from pathlib import Path
vals = [l.split("=", 1)[1].strip().strip("\"'") for l in Path(".env").read_text().splitlines()
        if "=" in l and not l.lstrip().startswith("#")]
vals = [v.encode() for v in vals if len(v) >= 4]
hits = [str(p) for p in Path("outputs/web_static").rglob("*") if p.is_file() and any(v in p.read_bytes() for v in vals)]
print("STOP: .env values found in", hits) if hits else print("OK: no .env values in the snapshot")
EOF
```

## 4. Where each part lives

| Part | Recommended home | Why |
|---|---|---|
| Code | GitHub (this repository) | Already there; versioned |
| Dataset files (raw ZIPs, clean parquet, tables, docs, checksums) | **Zenodo** record | Free; DOI for the paper; 50 GB and 100 files per record; versioned releases. You can reserve the DOI in a draft and publish the record with the paper |
| Website | Static host (below), downloads pointing at Zenodo | Free; no server to maintain; no credentials online |

## 5. Hosting options compared

Checked October 2026. Limits change, so confirm them on each provider's page before deploying.

| Option | Cost | Can it stay hidden before publication? | Limits that matter here | Live refresh / portal check |
|---|---|---|---|---|
| **GitHub Pages** (static) | Free for public repositories | **No.** A deployed Pages site is public. Publishing from a private repository needs a paid plan, and the site is still public | Site ≤ 1 GB, soft 100 GB/month bandwidth, 10-minute deploy timeout | No |
| **Cloudflare Pages** (static) | Free plan | **Yes.** Put Cloudflare Access (Zero Trust free plan, up to 50 users) in front of `*.pages.dev`; viewers sign in with an emailed one-time PIN | 20,000 files and 25 MiB per file on the free plan; no bandwidth cap is listed | No |
| **Render** static site | Free | No built-in password | Shared monthly bandwidth allowance | No |
| **Render** web service (live mode) | Free tier exists | Only if you add a login | Free services sleep after 15 minutes without traffic and take about a minute to wake; 750 free instance hours a month; ephemeral disk, so the bundle must be rebuilt on every deploy | Yes, but the portal credentials would have to be stored on Render |
| University hosting | Ask UW IST / WISA | Depends | Depends | Depends |

**Recommendation**

1. **Now:** run the site locally (section 2). If co-authors need to see it, use Cloudflare Pages behind Cloudflare Access with their email addresses. It costs nothing, but it creates a deployment, so it needs your go-ahead.
2. **At publication:**
   - publish the Zenodo record;
   - build the snapshot with `--downloads-url` pointing at the record;
   - deploy it to Cloudflare Pages (remove the Access policy) or to GitHub Pages;
   - put the URL and DOI in the paper's Data Availability section.
3. **Do not** run the live portal check on a public server. It needs your portal credentials on that server, and the public site gains nothing from it: the published dataset is a fixed, versioned release.

### Deploy commands (for when you approve)

Cloudflare Pages, direct upload (needs a free Cloudflare account):

```bash
npx wrangler pages deploy outputs/web_static --project-name eplanes
```

GitHub Pages: in a repository you choose for the site, upload the contents of `outputs/web_static/` to the branch or folder Pages serves. Alternatively, use the `actions/upload-pages-artifact` and `actions/deploy-pages` workflow. Keep the site in a separate repository or branch from the code if you want the code repository to stay data-free.

## 6. Maps (no API key)

The basemap is **OpenFreeMap**:

- no API key, no registration and no request limits on its public instance;
- commercial use allowed, and free;
- attribution required. MapLibre shows it automatically: "OpenFreeMap © OpenMapTiles Data from OpenStreetMap".

The site uses `https://tiles.openfreemap.org/styles/dark` (Night) and `.../styles/positron` (Day). If the tiles cannot be reached, the tracks are drawn on a plain background with a note, so the site never breaks.

To use another style, add these keys to `config.js` in the built site:

```js
window.EPLANES_CONFIG = {mode: 'static', api: 'data/', map_style_dark: '<style URL>', map_style_light: '<style URL>'};
```

Possible styles:
- OpenFreeMap's other styles;
- a self-hosted PMTiles style on the same host, for no third-party requests at all.

Providers such as MapTiler, Stadia Maps and CARTO now ask for an API key, which in a static site would be visible to everyone. Avoid them unless the key is locked to your domain.

## 7. Logos and photos

Only official logo files are used, never redrawn ones. Put the files into `eplanes/web/static/img/logos/` with these names:

| Organisation | Dark theme file | Light theme file |
|---|---|---|
| University of Waterloo | `uwaterloo.png` | `uwaterloo-light.png` |
| Waterloo Institute for Sustainable Aeronautics | `wisa.png` | `wisa-light.png` |
| Region of Waterloo International Airport | `waterloo-airport.png` | `waterloo-airport-light.png` |

Until a file exists, the footer shows the organisation's name as text. Use of each logo needs that organisation's permission. Photos are listed in `img/site.json` with a credit that is always displayed.

## 8. Checklist before the site goes public

- [ ] Redistribution permission from the operators (both fleets) and, if required, the manufacturer
- [ ] Decision on GPS tracks and registrations in the public release
- [ ] Licence chosen (e.g. CC BY 4.0) and added to the release README and the site
- [ ] Zenodo record published; DOI in the paper and on the site
- [ ] Logo permissions and official files
- [ ] Pipeline re-run from a clean state: `quality`, `export-web`, `release`, `web-static`
- [ ] `.env` scan of `outputs/web_static` (section 3) passes, and `git status` shows no data files
- [ ] Your explicit go-ahead to deploy
