"""No credentials or data in tracked-style files."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "data", "data_synthetic", "outputs", "outputs_synthetic", "__pycache__", ".pytest_cache", ".venv"}
PAT = re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"\s]{4,}['\"]")


def _files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP) and p.suffix in {".py", ".md", ".yaml", ".yml", ".txt", ".toml", ".json"}:
            if p.name == "test_repo_hygiene.py":
                continue
            yield p


def test_no_hardcoded_passwords():
    bad = [str(p) for p in _files() if PAT.search(p.read_text(errors="ignore"))]
    assert not bad, bad


def test_env_files_ignored():
    gi = (ROOT / ".gitignore").read_text()
    assert ".env" in gi and "data/" in gi and "*.zip" in gi
