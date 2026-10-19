"""Configuration loading and canonical column names."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# raw CSV header (as served by the portal) -> canonical snake_case name.
# Unknown headers are kept too: see eplanes.build.raw.slugify.
RAW_TO_CANON = {
    "time(min)": "time_min",
    "bat 1 min cell volt": "bat_1_min_cell_volt",
    "bat 1 max cell volt": "bat_1_max_cell_volt",
    "bat 2 min cell volt": "bat_2_min_cell_volt",
    "bat 2 max cell volt": "bat_2_max_cell_volt",
    "bat 1 soc": "bat_1_soc",
    "bat 2 soc": "bat_2_soc",
    "bat 1 soh": "bat_1_soh",
    "bat 2 soh": "bat_2_soh",
    "bat 1 current": "bat_1_current",
    "bat 2 current": "bat_2_current",
    "bat 1 max cell temp": "bat_1_max_cell_temp",
    "bat 2 max cell temp": "bat_2_max_cell_temp",
    "bat 1 avg cell temp": "bat_1_avg_cell_temp",
}

# Signals analysed per battery (suffix after bat_<k>_)
BAT_SIGNALS = ["min_cell_volt", "max_cell_volt", "soc", "soh", "current",
               "max_cell_temp", "avg_cell_temp"]

SIGNAL_DESCRIPTIONS = {
    "time_min": ("Time since start of the recording", "min"),
    "min_cell_volt": ("Lowest cell voltage in the pack", "mV (inferred)"),
    "max_cell_volt": ("Highest cell voltage in the pack", "mV (inferred)"),
    "soc": ("BMS-reported state of charge (integer-quantised)", "%"),
    "soh": ("BMS-reported state of health (NOT a measured capacity)", "%"),
    "current": ("Pack current; negative = charging", "A"),
    "max_cell_temp": ("Highest cell temperature in the pack", "degC"),
    "avg_cell_temp": ("Mean cell temperature (present for battery 1 only)", "degC"),
}


@dataclass
class Settings:
    raw: dict
    root: Path

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def data_dir(self) -> Path:
        return self.root / self.raw["paths"]["data_dir"]

    @property
    def outputs_dir(self) -> Path:
        return self.root / self.raw["paths"]["outputs_dir"]

    def with_dirs(self, data_dir: str | Path, outputs_dir: str | Path) -> "Settings":
        raw = dict(self.raw)
        raw["paths"] = {"data_dir": str(data_dir), "outputs_dir": str(outputs_dir)}
        return Settings(raw, Path("/"))  # absolute paths expected


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Settings(raw, ROOT)


ENV_FILE = ROOT / ".env"


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Minimal .env parser (no python-dotenv needed). Accepts KEY=value, `export KEY=value`,
    optional quotes, spaces around '=', comments, a UTF-8 BOM and Windows line endings."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        out[k] = v
    return out


def credentials(account: str, settings: Settings) -> tuple[str, str]:
    """Read credentials from the process environment or ROOT/.env. Never from code."""
    acct = settings["accounts"][account]
    env = read_env_file()
    user = os.environ.get(acct["user_env"]) or env.get(acct["user_env"], "")
    pw = os.environ.get(acct["password_env"]) or env.get(acct["password_env"], "")
    if not user or not pw:
        if not ENV_FILE.exists():
            nearby = sorted(p.name for p in ROOT.iterdir() if "env" in p.name.lower())
            where = f"{ENV_FILE} does not exist (files with 'env' in the name here: {nearby or 'none'})"
        else:
            where = f"{ENV_FILE} exists and defines keys: {sorted(env) or 'none'}"  # names only, never values
        raise RuntimeError(
            f"Missing credentials for {account}: need non-empty {acct['user_env']} and {acct['password_env']}. "
            f"{where}. See .env.example."
        )
    return user, pw
