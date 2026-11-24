#!/usr/bin/env python3
"""Apply a SIGNED offline-update bundle (CUSTOMER side, on the air-gapped target).

Order is the whole point — nothing is loaded until the bundle is proven authentic AND a genuine
step forward:

    1. verify the Ed25519 SIGNATURE with the pre-installed Acme release public key
       (RAG_UPDATE_RELEASE_PUBLIC_KEY) — a swapped/forged bundle stops HERE;
    2. verify every artifact's SHA-256 against the (signed) manifest;
    3. check_upgrade — refuse a downgrade/replay or a jump that skips a required base;
    4. docker load images.tar  ->  alembic upgrade head  ->  compose up -d --force-recreate;
    5. poll /readyz; on failure ROLL BACK to the previous images and exit non-zero.

    uv run python scripts/apply_update.py --bundle ./update-b3 \
        --compose infra/compose/docker-compose.dev.yml --version-file /var/lib/rag/BUILD

The current build number lives in --version-file (0 if absent). It is advanced ONLY after a
green health check, so a failed/rolled-back update leaves the recorded build untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from rag_common import get_settings
from rag_common.update_manifest import (
    IncompatibleUpdateError,
    UpdateManifest,
    check_upgrade,
    verify_artifacts,
    verify_manifest,
)

_HTTP_OK = 200


def _readyz(url: str, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 — operator-supplied localhost URL
                if r.status == _HTTP_OK:
                    return True
        except OSError:
            pass
        time.sleep(3)
    return False


def _current_images(compose: str) -> list[str]:
    out = subprocess.run(
        ["docker", "compose", "-f", compose, "config", "--images"], check=True, capture_output=True, text=True
    ).stdout
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply a signed offline-update bundle.")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--compose", required=True)
    ap.add_argument("--version-file", required=True, help="File tracking the running build number.")
    ap.add_argument("--health-url", default="http://localhost:8000/readyz")
    ap.add_argument("--health-timeout", type=int, default=120)
    ap.add_argument("--public-key", default="", help="Override the release pubkey (else from config).")
    ap.add_argument("--insecure", action="store_true", help="Skip signature check (NEVER in production).")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    manifest = UpdateManifest(**json.loads((bundle / "manifest.json").read_text()))

    # 1. authenticity.
    pub = args.public_key or get_settings().update_release_public_key
    if not args.insecure:
        sig = (bundle / "manifest.sig").read_text().strip()
        if not pub:
            print("REFUSED: no release public key configured (RAG_UPDATE_RELEASE_PUBLIC_KEY) and not --insecure")
            return 3
        if not verify_manifest(manifest, sig, pub):
            print("REFUSED: signature invalid — this bundle is not a genuine Acme release")
            return 3
    # 2. integrity.
    bad = verify_artifacts(manifest, bundle)
    if bad:
        print(f"REFUSED: artifact checksum mismatch: {bad}")
        return 3
    # 3. monotonic step forward.
    vfile = Path(args.version_file)
    current = int(vfile.read_text().strip()) if vfile.is_file() else 0
    try:
        check_upgrade(current, manifest)
    except IncompatibleUpdateError as exc:
        print(f"REFUSED: {exc}")
        return 3

    print(f"verified: {manifest.bundle_id} build {current} -> {manifest.build_number}. applying...")
    rollback_images = _current_images(args.compose)
    try:
        subprocess.run(["docker", "load", "-i", str(bundle / "images.tar")], check=True)
        # migrations run inside the freshly loaded api image, then recreate the stack.
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                args.compose,
                "run",
                "--rm",
                "api",
                "alembic",
                "-c",
                "libs/clients/alembic.ini",
                "upgrade",
                "head",
            ],
            check=True,
        )
        subprocess.run(["docker", "compose", "-f", args.compose, "up", "-d", "--force-recreate"], check=True)
        if not _readyz(args.health_url, args.health_timeout):
            raise RuntimeError("post-update readiness check failed")
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"FAILED: {exc} — rolling back to the previous images")
        subprocess.run(
            ["docker", "compose", "-f", args.compose, "up", "-d", "--force-recreate"],
            check=False,
            env={"RAG_ROLLBACK_IMAGES": ",".join(rollback_images)},
        )
        return 1

    vfile.parent.mkdir(parents=True, exist_ok=True)
    vfile.write_text(str(manifest.build_number))
    print(f"UPDATED to build {manifest.build_number} ({manifest.product_version}). health OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
