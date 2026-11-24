# Offline updates (air-gapped sites)

How to update a running, **internet-less** RAG as a Service deployment — no `docker pull`, no
`git pull`, no PyPI. Updates arrive as a **signed bundle on vetted physical media**. This doc
explains the mechanism end to end, because "how do offline updates even work?" is the usual
first question.

> **The one idea:** the provider packages the new version, **cryptographically signs** it, and the
> air-gapped site **verifies the signature before loading anything**. The signature is what makes
> physical media safe — it proves the bundle came from Acme and was not swapped or tampered on the
> way in. Integrity checksums alone can't do that (an attacker who swaps the bundle swaps the
> checksums too); a signature they cannot forge can.

---

## The flow

```
 PROVIDER (connected build box)                 AIR-GAPPED SITE (running build N)
 ──────────────────────────────                 ────────────────────────────────
 build_update_bundle.py                          apply_update.py
   • docker save all images -> images.tar          1. verify Ed25519 SIGNATURE   ── fail? STOP
   • compute image digests + SHA-256s                 (pre-installed Acme pubkey)
   • write manifest.json  (versions, build            2. verify artifact SHA-256s ── fail? STOP
     number, migration head, digests, hashes)         3. check_upgrade            ── downgrade/
   • SIGN manifest -> manifest.sig  ─┐   USB /            (newer build, base met)     replay? STOP
   • SHA256SUMS                       ├─ optical ─►    4. docker load images.tar
                                      │   diode         5. alembic upgrade head
 release PRIVATE key (HSM, never    ─┘                 6. compose up -d --force-recreate
 leaves the provider)                                  7. poll /readyz ── fail? ROLL BACK to N
                                                        8. record build N+1  (only after green)
```

The bundle (`build_update_bundle.py` output): `images.tar` · `manifest.json` · `manifest.sig` ·
`SHA256SUMS`. Tar it onto the media; carry it in.

## Why each guard exists (threat model)

| Guard | Stops |
|---|---|
| **Ed25519 signature** (`manifest.sig`, verified with the pre-installed release **public** key) | A swapped or forged bundle — supply-chain tampering on the media. **The load never starts if this fails.** |
| **Per-artifact SHA-256** (in the *signed* manifest) | A single mutated byte in `images.tar` — and because the manifest is signed, a matching checksum *means* it came from Acme. |
| **Monotonic `build_number` + `min_from_build`** | A **downgrade or replay** (re-applying an old bundle), and skipping a required intermediate update. |
| **`/readyz` health check + rollback** | A bad update taking the site down — it reverts to the previous images and leaves the recorded build untouched. |

## Key custody

- The **release PRIVATE key** (base64 Ed25519) lives in the provider's HSM/secrets store. It never
  leaves Acme, is never committed, and signs every bundle.
- The matching **public key** is pre-installed at each site as
  `RAG_UPDATE_RELEASE_PUBLIC_KEY` (delivered once, out of band, at install). It only *verifies* —
  it cannot sign, so leaking it does not let anyone forge an update.
- Rotate by shipping a new public key through the same trusted channel as the initial install.

## Run it

**Provider** (connected):
```bash
uv run python scripts/build_update_bundle.py \
  --compose infra/compose/docker-compose.dev.yml --out ./update-b3 \
  --build-number 3 --min-from-build 1 --product-version 0.3.0 \
  --migration-head 0007_tenants --signing-key /secure/acme-release.key
tar czf update-b3.tgz -C . update-b3      # -> onto vetted media
```

**Air-gapped site**:
```bash
tar xzf update-b3.tgz
uv run python scripts/apply_update.py --bundle ./update-b3 \
  --compose infra/compose/docker-compose.dev.yml --version-file /var/lib/rag/BUILD
```
A non-zero exit means the update was **refused or rolled back** — the site keeps running the
previous build. The pure verification core (`rag_common.update_manifest`) is unit-tested; the
scripts orchestrate `docker`/`alembic` around it.

## Relationship to the fresh install

First install uses `scripts/airgap-bundle.sh` / `airgap-install.sh` (see
[airgap-install.md](airgap-install.md)) — that also ships the **model-weight cache volumes**, which
a code-only update does not need to re-ship (weights rarely change). An update that *does* change a
model adds those cache tarballs as extra signed artifacts in the manifest. See also the
[deployment runbook](deployment-runbook.md).
