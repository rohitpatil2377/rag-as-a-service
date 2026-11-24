#!/usr/bin/env python3
"""Build a SIGNED offline-update bundle (PROVIDER side, on a connected build box).

Produces a directory you tar onto vetted physical media and carry to an air-gapped site:

    images.tar      docker save of every image the compose stack uses
    manifest.json   versions + build number + migration head + image digests + artifact SHA-256s
    manifest.sig    detached Ed25519 signature of manifest.json (Acme release key)
    SHA256SUMS      plain checksums (compat with the existing air-gap installer)

    uv run python scripts/build_update_bundle.py \
        --compose infra/compose/docker-compose.dev.yml --out ./update-b3 \
        --build-number 3 --min-from-build 1 --product-version 0.3.0 \
        --migration-head 0007_tenants --signing-key /secure/acme-release.key

The signing key is a base64 raw Ed25519 private key kept in the provider's HSM/secrets store —
NEVER on the target, never committed. The matching PUBLIC key is pre-installed at each site as
RAG_UPDATE_RELEASE_PUBLIC_KEY so it can verify what this produces.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from rag_common.update_manifest import Artifact, ImageRef, UpdateManifest, sha256_file, sign_manifest


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def _images(compose: str) -> list[str]:
    out = _run(["docker", "compose", "-f", compose, "config", "--images"])
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def _digest(ref: str) -> str:
    try:
        raw = _run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", ref])
        return raw.split("@", 1)[1] if "@" in raw else ""
    except subprocess.CalledProcessError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a signed offline-update bundle.")
    ap.add_argument("--compose", required=True)
    ap.add_argument("--out", required=True, help="Bundle output directory (created).")
    ap.add_argument("--build-number", type=int, required=True)
    ap.add_argument("--min-from-build", type=int, required=True)
    ap.add_argument("--product-version", required=True)
    ap.add_argument("--migration-head", required=True)
    ap.add_argument("--signing-key", required=True, help="Path to the base64 Ed25519 release PRIVATE key.")
    ap.add_argument("--bundle-id", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = _images(args.compose)
    if not images:
        print("no images resolved from the compose file")
        return 2

    print(f"saving {len(images)} images -> images.tar")
    images_tar = out / "images.tar"
    subprocess.run(["docker", "save", "-o", str(images_tar), *images], check=True)

    manifest = UpdateManifest(
        bundle_id=args.bundle_id or f"acme-{args.build_number}",
        product_version=args.product_version,
        build_number=args.build_number,
        min_from_build=args.min_from_build,
        created_at=datetime.now(UTC).isoformat(),
        migration_head=args.migration_head,
        images=[ImageRef(ref=r, digest=_digest(r)) for r in images],
        artifacts=[Artifact(path="images.tar", sha256=sha256_file(images_tar), bytes=images_tar.stat().st_size)],
    )
    (out / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2))
    signing_key = Path(args.signing_key).read_text().strip()
    (out / "manifest.sig").write_text(sign_manifest(manifest, signing_key))
    (out / "SHA256SUMS").write_text(f"{sha256_file(images_tar)}  images.tar\n")

    print(f"bundle ready: {out}  (build {args.build_number}, signed)")
    print("tar it onto vetted media:  tar czf update.tgz -C", out.parent, out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
