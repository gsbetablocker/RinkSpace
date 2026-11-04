"""Verify/apply the camera-only patch to an existing, pinned external ExMR checkout.

Default is verification only. Pass --apply to change exmr/model/exmr.py.
No source, package or weight downloads, inference, commits or resets are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


COMMIT = "eac72e47339b404c0443f5fb323412e2ce432410"
RELATIVE_SOURCE = "exmr/model/exmr.py"
ORIGINAL_SHA256 = "6faf5964522dad01c5ee83d82704cb66de7a4601e0db375a9aa6496b263b83db"
PATCHED_SHA256 = "af2c8af12fa0e315c4c046b1643f29b54de7dcda4409b67a23e1fb0584cbdd20"
PATCH_SHA256 = "7c31b88b2f34b5e1ffe65ab08eaeb1e4677aad82ff33bf650d957fa6b51a71ac"
PATCH = Path(__file__).with_name("exmr_camera.patch")
PROJECT = Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _source_sha(path: Path) -> str:
    # Git's Windows checkout may use CRLF; identity is the pinned LF source text.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def prepare(source_root: Path, *, apply: bool = False) -> dict:
    root = Path(source_root).resolve(strict=True)
    if root.is_relative_to(PROJECT) or PROJECT.is_relative_to(root):
        raise ValueError("ExMR source must be in external Data, separate from OneDrive project")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("source_root must be the ExMR Git checkout root")
    if _git(root, "rev-parse", "HEAD") != COMMIT:
        raise ValueError(f"ExMR HEAD must be pinned to {COMMIT}")
    target = (root / RELATIVE_SOURCE).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError("ExMR source member escapes checkout")
    if hashlib.sha256(PATCH.read_bytes()).hexdigest() != PATCH_SHA256:
        raise ValueError("Camera patch SHA256 mismatch; preserved checkout without changes")
    identity = _source_sha(target)
    if identity == PATCHED_SHA256:
        _git(root, "apply", "--reverse", "--check", "--", str(PATCH))
        status = "already_applied"
    elif identity == ORIGINAL_SHA256:
        _git(root, "apply", "--check", "--", str(PATCH))
        status = "ready"
        if apply:
            _git(root, "apply", "--", str(PATCH))
            if _source_sha(target) != PATCHED_SHA256:
                raise RuntimeError("Applied source SHA256 differs from the reviewed patch")
            status = "applied"
    else:
        raise ValueError("ExMR source SHA256 mismatch; preserved local changes without overwrite")
    return dict(status=status, commit=COMMIT, source=str(target),
                source_sha256=_source_sha(target), patch_sha256=PATCH_SHA256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="External Data/runtime_tools/body/ExMR Git checkout")
    parser.add_argument("--apply", action="store_true", help="Apply only to the exact pinned source")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, apply=args.apply)))


if __name__ == "__main__":
    main()
