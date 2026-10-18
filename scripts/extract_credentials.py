#!/usr/bin/env python
"""Create a local .env from credentials already present in your notebooks.

Usage:  python scripts/extract_credentials.py path/to/EDA_Data.ipynb [more.ipynb ...]

It looks for `USERNAME = '...'` / `PASSWORD = '...'` pairs, maps the user names to
Fleet A / Fleet B using --fleet-a / --fleet-b (the portal user names), and writes
./.env with mode 600. Nothing is printed except which users were found.
After this, REMOVE passwords from the notebooks (and rotate them if the notebooks
were ever shared or pushed).
"""
import argparse
import json
import os
import re
from pathlib import Path

PAIR = re.compile(r"USERNAME\s*=\s*['\"]([^'\"]+)['\"].*?PASSWORD\s*=\s*['\"]([^'\"]+)['\"]", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--fleet-a", required=True, help="portal user name for the C-GAUW/C-GMUW account")
    ap.add_argument("--fleet-b", required=True, help="portal user name for the VSW1280102/VSW1280107 account")
    a = ap.parse_args()
    found = {}
    for nb in a.notebooks:
        for c in json.load(open(nb))["cells"]:
            if c["cell_type"] != "code":
                continue
            for u, p in PAIR.findall("".join(c["source"])):
                found[u] = p
    out = []
    for key, user in (("A", a.fleet_a), ("B", a.fleet_b)):
        if user in found:
            out += [f"PIPISTREL_FLEET_{key}_USER={user}", f"PIPISTREL_FLEET_{key}_PASSWORD={found[user]}"]
            print(f"found credentials for {user} -> fleet {key}")
        else:
            print(f"NOT found: {user}")
    env = Path(__file__).resolve().parents[1] / ".env"
    env.write_text("\n".join(out) + "\n")
    os.chmod(env, 0o600)
    print(f"wrote {env} (git-ignored)")


if __name__ == "__main__":
    main()
