# Model SBOM

A **Model SBOM** (Software Bill of Materials for AI weights) is a signed manifest that states,
precisely and checkably, which model weights a deployment runs. It backs the platform's
**"Complete Indigenous / self-hosted models"** sovereignty claim: instead of asking a customer to
trust a marketing slide, the provider signs an attestation the customer can verify offline.

Implementation: `libs/common/src/rag_common/model_sbom.py`. Same pattern as the offline-update
manifest (`rag_common.update_manifest`): canonical sorted-keys JSON, detached Ed25519 signature,
fail-closed verification.

## What it asserts

For every model in the deployment, the SBOM records:

| field           | meaning                                                         |
|-----------------|------------------------------------------------------------------|
| `name`          | model identifier, e.g. `bge-m3`                                  |
| `role`          | one of `embed`, `rerank`, `generate`, `ocr`                      |
| `source_origin` | where the weights come from, e.g. `India` or a vendor name        |
| `license`       | the weight license, e.g. `MIT`, `Apache-2.0`                     |
| `weight_sha256` | SHA-256 of the weight file, hex-encoded                          |
| `revision`      | optional version/tag string                                      |

The SBOM as a whole also carries a `provenance` statement (default:
`"India-origin, self-hosted"`), a `deployment_id`, a `generated_at` timestamp, the `signer_kid`
identifying which release key signed it, and the `signed_token` (the signature itself).

A tampered SBOM — any weight hash, license, or origin changed after signing — fails verification.
An unsigned SBOM, or one signed with the wrong key, also fails. `verify_sbom` never raises; it
returns `False` on any problem, so a caller can always treat "not `True`" as "not trustworthy."

## Building and signing (provider)

1. Write a `models.json` describing every model in the deployment — see
   `scripts/build_model_sbom.py` for the exact shape, or the example below.
2. Keep the release Ed25519 **private** key off the target — in an HSM or secrets store, as a
   base64 raw-key file, never committed.
3. Run:

   ```bash
   python scripts/build_model_sbom.py models.json /secure/acme-release.key -o model-sbom.json
   ```

   This hashes every `weight_path` on disk (`entry_for_file`), builds the `ModelSBOM`, signs it
   (`sign_sbom`), and writes the signed JSON to `model-sbom.json`. Omit `-o` to print to stdout.

Example `models.json`:

```json
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
    },
    {
      "name": "krutrim-7b",
      "role": "generate",
      "source_origin": "India",
      "license": "Apache-2.0",
      "weight_path": "/models/krutrim-7b/model.safetensors"
    }
  ]
}
```

`provenance` is optional (defaults to `"India-origin, self-hosted"`); each model's `revision` is
optional (defaults to `""`).

## Verifying (customer / auditor)

Given the signed `model-sbom.json` and the release **public** key (pre-shared out of band, the
same key used to verify offline-update bundles):

```python
import base64
import json

from rag_common.model_sbom import ModelSBOM, verify_sbom

sbom = ModelSBOM.model_validate(json.loads(open("model-sbom.json").read()))
public_key_b64 = open("acme-release-public.key").read().strip()

assert verify_sbom(sbom, public_key_b64) is True  # False => do not trust this SBOM
```

To confirm the weights actually deployed match the SBOM (not just that the SBOM is authentically
signed), recompute the SHA-256 of each deployed weight file and compare against the corresponding
`ModelEntry.weight_sha256` — the same recompute-and-compare pattern
`rag_common.update_manifest.verify_artifacts` uses for update bundles.

## How this backs the sovereignty claim

"India-origin, self-hosted models" is a claim about supply chain and deployment topology that a
UI screenshot cannot prove. The Model SBOM makes it falsifiable:

- **Origin** — `source_origin` on every model, so a customer can see exactly which weights are
  claimed as India-origin vs. third-party, per model rather than as one blanket statement.
- **Integrity** — `weight_sha256` ties the claim to the literal bytes running in production; a
  swapped or downgraded model changes the hash and breaks the match.
- **Authenticity** — the Ed25519 signature ties the whole document to the provider's release key,
  so the SBOM itself can't be forged or silently edited in transit.

A customer or regulator can therefore audit "which models does this deployment actually run, and
were they claimed as India-origin at signing time" without depending on the provider's continued
cooperation.
