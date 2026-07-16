from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_relative_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    local_targets = [target.split("#", 1)[0] for target in targets if "://" not in target]

    missing = [target for target in local_targets if target and not (ROOT / target).exists()]
    assert missing == []


def test_current_docs_do_not_reference_deleted_gitee_guide() -> None:
    current_docs = [ROOT / "README.md", ROOT / "START_HERE.md", ROOT / "docs" / "project_structure.md"]

    assert all("gitee_sync.md" not in path.read_text(encoding="utf-8") for path in current_docs)
