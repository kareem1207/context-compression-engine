"""The version is stated in three places; they must agree."""

import json
import tomllib
from pathlib import Path

from cce_rest.app import create_app

ROOT = Path(__file__).resolve().parent.parent


def test_python_rest_and_npm_versions_agree():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    npm = json.loads((ROOT / "cce_sdk_js" / "package.json").read_text(encoding="utf-8"))["version"]
    assert create_app().version == py == npm


def test_changelog_has_an_entry_for_this_version():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    first = next(l for l in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines() if l.startswith("## "))
    assert first.startswith(f"## {py} (") and "unreleased" not in first
