# Tests

Tests live next to the code they cover (`libs/*/tests/`, `services/*/tests/`). This directory holds
only the cross-cutting suites that exercise the assembled platform.

| Marker | Where | Count | Infrastructure | In `make ci`? |
|---|---|---|---|---|
| *(unmarked)* | `libs/`, `services/` | 2,947 | none | yes |
| `e2e` | `tests/e2e/` (93 files) | 394 | none — in-memory profile | **no**, run separately |
| `integration` | `tests/integration/` (15 files) | 114 | needs `make up` | no |
| `eval` | gold suites | — | needs the real cross-encoder | no |

```bash
make ci                                        # ruff + mypy strict + the 2,947 unit tests
uv run pytest -q -m e2e                        # 394, no infrastructure
make up && uv run pytest -q -m integration     # 114, live stores (see README for the env)
```

**A green `make ci` is not a full pass.** It excludes e2e, which needs nothing but is still
excluded, so run it before claiming work is done. CI runs it as its own step.

## The one test that matters most

`tests/integration/test_tenant_isolation.py` proves tenant B cannot read tenant A's data across
*every* store — vector, BM25, object and catalog. Its adversarial sibling
`test_tenant_isolation_adversarial.py` attacks the same boundary through the API: forged tokens,
mismatched API keys, cross-tenant KB ids. Treat a failure in either as release-blocking.

## Writing an integration test

It runs against a **persistent** Postgres, so it must be idempotent: a hardcoded primary key passes
on a virgin database and fails on every run after. Derive ids per run, or clean up. Two tests here
had exactly that bug and only revealed it when the suite was run twice in a row — which is now the
standard for calling the real-store pass green.
