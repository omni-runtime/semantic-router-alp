#!/usr/bin/env python3
"""Check public sources, dependency boundaries, links and patch provenance."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".work", ".venv", ".pytest_cache", "__pycache__", "build", "dist", "reports", "results"}


def main():
    for name in ("README.md", "README.zh-CN.md", "LICENSE", "NOTICE", "CONTRIBUTING.md",
                 "SECURITY.md", "CODE_OF_CONDUCT.md", "CHANGELOG.md", ".github/workflows/ci.yml"):
        assert (ROOT / name).is_file(), name
    manifest = json.loads((ROOT / "patches/checksums.json").read_text())
    series = (ROOT / "patches/series").read_text().splitlines()
    assert len(series) == len(set(series)) and set(series) == set(manifest)
    for name in series:
        assert Path(name).name == name
        assert hashlib.sha256((ROOT / "patches" / name).read_bytes()).hexdigest() == manifest[name]
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    for key in ("upstream", "multimodal", "contracts"):
        assert re.fullmatch(r"[0-9a-f]{40}", lock[key]["commit"]), key
    assert re.fullmatch(r"[0-9a-f]{64}", lock["composed_patch_sha256"])
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert f"alp-schema-mcp[runtime]=={lock['contracts']['package_version']}" in project["dependencies"]
    assert not any("vllm" in d or "xgrammar" in d for d in project["dependencies"])
    broken = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or EXCLUDED.intersection(path.relative_to(ROOT).parts):
            continue
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text()):
                if re.match(r"(?:https?://|mailto:|#)", target):
                    continue
                if not (path.parent / target.split("#", 1)[0]).exists():
                    broken.append(f"{path.relative_to(ROOT)} -> {target}")
        if path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
        if path.suffix == ".py":
            tree = ast.parse(path.read_text())
            if "src" in path.relative_to(ROOT).parts:
                for node in ast.walk(tree):
                    names = [n.name for n in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                    assert not any(n.split(".")[0] in {"vllm_alp", "vllm", "mlx", "torch", "xgrammar"} for n in names), path
    assert not broken, "\n".join(broken)
    assert not (ROOT / "src/semantic_router_alp/contracts").exists()
    print("Public structure, links, syntax, dependency boundary and patch provenance passed.")


if __name__ == "__main__":
    main()
