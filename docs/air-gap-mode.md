# Air-gap mode

Sovereign mode (`sovereign_enforce` + `allowed_egress_hosts`) allow-lists which outbound
hosts a deployment may reach. It does not promise a deployment reaches none. Air-gap mode
is the stricter contract for sites with no outbound connectivity at all: turning it on
means the platform has verified, at startup, that nothing in its own configuration would
try to dial out.

## What it guarantees

When `air_gap_enabled` is true, the platform will refuse to start if any of the following
are configured — each is a setting that implies a live outbound call with no offline
substitute:

| Check | Setting(s) | Rule |
|---|---|---|
| Hosted/external model or OCR backend | `llm_backend`, `embed_backend`, `rerank_backend`, `ocr_backend`, `vision_llm_backend` + their `*_url` | Flagged only if the backend is set to a hosted value (`openai_compat`, `chandra`, `chandra_api`) **and** the URL it actually dials resolves off-box. The same backend pointed at the operator's own intranet endpoint is fine. |
| SMTP relay | `smtp_host` | Any value is outbound email egress. Leave empty — air-gapped sites fall back to an admin-copied activation/reset link. |
| OIDC JWKS | `oidc_jwks_url` | Fetched live on every token verification; flagged whenever set. |
| Remote SIEM | `siem_enabled` + `siem_endpoint` | Only flagged when forwarding is enabled **and** the endpoint resolves off-box; an on-prem collector is fine. |
| Update verification key | `update_release_public_key` | Flagged when **empty**. An air-gapped site can never phone home to check an update bundle's provenance, so it must be able to verify bundles locally — shipping without that key is itself a risk. |

"Off-box" means: not loopback, not an RFC1918-private IP, not a bare intranet/docker
service name (no dot — e.g. `qdrant`, `vllm`), and not a `.local`/`.internal` hostname.
Everything else (a real public FQDN, e.g. `www.datalab.to`) counts as external.

## Turning it on

Set `RAG_AIR_GAP_ENABLED=true` (or the equivalent `air_gap_enabled` setting) for a fully
offline site. The platform runs the preflight checker at startup; if any violation is
found, startup fails fast with a `ValueError` naming every offending setting, rather than
starting and making a silent outbound call the first time a token is verified or an email
is sent.

## Reading the report

An ops preflight endpoint exposes the same checks as a report:

```json
{
  "air_gap_enabled": true,
  "clean": false,
  "violations": [
    {
      "setting": "smtp_host",
      "value": "smtp.gmail.com",
      "reason": "SMTP relay configured; email is outbound egress with no offline substitute"
    }
  ]
}
```

- `air_gap_enabled` — whether the flag is currently on.
- `clean` — whether the current configuration has zero violations.
- `violations` — every offending setting, its current value, and why it's a problem.

The report is computed regardless of whether `air_gap_enabled` is on, so an operator can
preview "would this deployment pass air-gap" before flipping the flag.
