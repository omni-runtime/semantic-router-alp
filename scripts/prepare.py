#!/usr/bin/env python3
"""Compose locked upstream, multimodal and ALP sources in a new checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*argv, cwd=None):
    subprocess.run(argv, cwd=cwd, check=True)


def show(repo, revision, name):
    return subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=repo)


def checked_patches(series, manifest, read):
    names = [n for n in series.splitlines() if n and not n.startswith("#")]
    if len(names) != len(set(names)) or set(names) != set(manifest):
        raise ValueError("Patch series/checksum mismatch")
    result = []
    for name in names:
        if Path(name).name != name:
            raise ValueError("Patch names must be basenames")
        raw = read(name)
        if hashlib.sha256(raw).hexdigest() != manifest[name]:
            raise ValueError(f"Patch checksum mismatch: {name}")
        result.append(raw)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", help="Existing upstream Git clone (optional)")
    p.add_argument("--multimodal", type=Path, help="Existing multimodal Git clone (optional)")
    p.add_argument("--output", type=Path, default=ROOT / ".work/prepared")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a new directory")
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    base, upstream = lock["multimodal"], lock["upstream"]
    with tempfile.TemporaryDirectory(prefix="sr-alp-compose-") as temp:
        temp = Path(temp)
        repo = args.multimodal
        if repo is None:
            repo = temp / "multimodal"
            run("git", "clone", "--no-checkout", base["repository"], str(repo))
        base_lock = json.loads(show(repo, base["commit"], "upstream.lock.json"))
        if base_lock != upstream:
            raise ValueError("Multimodal upstream lock differs from ALP lock")
        manifest = json.loads(show(repo, base["commit"], "patches/checksums.json"))
        if manifest != base["patches"]:
            raise ValueError("Multimodal patch manifest differs from ALP lock")
        patches = checked_patches(
            show(repo, base["commit"], "patches/series").decode(), manifest,
            lambda n: show(repo, base["commit"], "patches/" + n),
        )
        patches += checked_patches(
            (ROOT / "patches/series").read_text(),
            json.loads((ROOT / "patches/checksums.json").read_text()),
            lambda n: (ROOT / "patches" / n).read_bytes(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--no-hardlinks", "--no-checkout",
            args.source or upstream["repository"], str(args.output))
        run("git", "checkout", "--detach", upstream["commit"], cwd=args.output)
        for index, raw in enumerate(patches):
            path = temp / f"{index}.patch"
            path.write_bytes(raw)
            run("git", "apply", "--check", str(path), cwd=args.output)
            run("git", "apply", str(path), cwd=args.output)
        # Include new files in the source fingerprint without creating a commit.
        run("git", "add", "--intent-to-add", ".", cwd=args.output)
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=args.output)
        digest = hashlib.sha256(diff).hexdigest()
        if digest != lock["composed_patch_sha256"]:
            raise ValueError("Composed source differs from its locked fingerprint")
    print(json.dumps({"checkout": str(args.output.resolve()), "composed_patch_sha256": digest}))


if __name__ == "__main__":
    main()
