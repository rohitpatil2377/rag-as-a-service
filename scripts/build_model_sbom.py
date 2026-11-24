#!/usr/bin/env python3
"""Build + sign a Model SBOM (PROVIDER side): the "which weights actually run" attestation that
backs the "India-origin, self-hosted" sovereignty claim.

Given a JSON config describing the models in the deployment (name/role/origin/license/weight
path/revision), this hashes each weight file, builds a :class:`~rag_common.model_sbom.ModelSBOM`,
signs it with the release private key, and writes the signed SBOM JSON to stdout or ``-o``.

    python scripts/build_model_sbom.py models.json /secure/acme-release.key -o model-sbom.json

``models.json`` shape:

    {
      "sbom_id": "sbom-2026-08-05",
      "deployment_id": "acme-blr-01",
      "signer_kid": "acme-release-2026",
      "provenance": "India-origin, self-hosted",
      "models": [
        {
          "name": "bge-m3",
          "role": "embed",
          "source_origin": "India",
          "license": "MIT",
          "weight_path": "/models/bge-m3/model.safetensors",
          "revision": "v1"
        }
      ]
    }

``provenance`` is optional (defaults to "India-origin, self-hosted"). Each model's ``revision`` is
optional (defaults to ""). The signing key file holds a base64 raw Ed25519 PRIVATE key — kept in
the provider's HSM/secrets store, never on the target, never committed. The matching PUBLIC key is
what a customer/auditor uses to verify the output with
:func:`rag_common.model_sbom.verify_sbom`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag_common.model_sbom import ModelEntry, ModelSBOM, entry_for_file, sign_sbom


def _build_sbom(config: dict[str, object]) -> ModelSBOM:
    raw_models = config.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models.json must contain a non-empty 'models' list")

    entries: list[ModelEntry] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            raise ValueError(f"each model entry must be an object, got: {raw!r}")
        entries.append(
            entry_for_file(
                name=str(raw["name"]),
                role=str(raw["role"]),  # type: ignore[arg-type]
                source_origin=str(raw["source_origin"]),
                license=str(raw["license"]),
                weight_path=Path(str(raw["weight_path"])),
                revision=str(raw.get("revision", "")),
            )
        )

    kwargs: dict[str, object] = {
        "sbom_id": str(config["sbom_id"]),
        "deployment_id": str(config["deployment_id"]),
        "models": entries,
        "generated_at": datetime.now(UTC),
        "signer_kid": str(config["signer_kid"]),
    }
    if "provenance" in config:
        kwargs["provenance"] = str(config["provenance"])
    return ModelSBOM(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and sign a Model SBOM.")
    ap.add_argument("models_json", help="Path to the models config JSON (see module docstring for shape).")
    ap.add_argument("private_key_file", help="Path to a file holding the base64 Ed25519 release PRIVATE key.")
    ap.add_argument("-o", "--out", default="", help="Output path for the signed SBOM JSON (default: stdout).")
    args = ap.parse_args()

    config = json.loads(Path(args.models_json).read_text())
    sbom = _build_sbom(config)
    signing_key = Path(args.private_key_file).read_text().strip()
    signed = sign_sbom(sbom, signing_key)

    payload = json.dumps(signed.model_dump(mode="json"), indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n")
        print(f"signed model SBOM written to {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
