# Contributing

## Setup

```bash
make install   # uv sync — every lib, service, app and the dev toolchain
make hooks     # installs .githooks/pre-push, which runs `make ci` before each push
```

Requires [`uv`](https://docs.astral.sh/uv/) and Docker with Compose v2. Nothing else — the dev
stack needs no secrets, no model endpoint and no GPU.

## The gate

```bash
make ci                                      # ruff + mypy strict + 2,947 unit tests  (~1 min)
uv run pytest -q -m e2e                      # 394 tests, no infrastructure           (~2 min)
make up && uv run pytest -q -m integration   # 114 tests, live stores
```

**`make ci` alone is not a full pass** — it excludes the e2e suite. Run that too before saying a
change is done, and never describe a partial run as a full repository pass.

Do not substitute a path-scoped lint or typecheck for `make ci`. The scopes differ by whole
directories: `mypy libs services apps` omits `sdks/python`, and `ruff check libs services apps`
omits the repo-root `tests/`. Both gaps have let a defect through before, which is why the gate is
defined once in the Makefile and the CI workflow calls it rather than restating it.

## House rules

- **mypy strict, zero tolerance.** Keep it green.
- **`tenant_id` is an explicit parameter** on anything touching a store. Never infer it from
  ambient state for an isolation decision.
- **Additive and flag-gated.** A new behaviour ships behind a `RAG_*` flag that defaults to off,
  so the path with the flag off stays byte-identical.
- **No RAG framework.** `libs/rag-core` is Protocols plus pure logic on purpose
  ([ADR-0002](docs/adr/0002-thin-core-no-rag-framework.md)).
- **Services never import a sibling `svc_*`.** Cross-service work goes through a Protocol.
- **Read the ADR before reversing a decision that looks like an oversight.** It usually isn't one.
- **Record a new decision as an ADR** in `docs/adr/`, numbered in sequence.

## Tests

Put the test next to the code. Pure logic in `rag-core` can be asserted directly — if a change is
only observable through the API, it probably belongs in a different layer.

Integration tests run against a **persistent** Postgres, so they must be idempotent: run the suite
twice in a row and it must pass both times. A hardcoded primary key passes once and fails forever
after.

## Commits

Explain *why* in the body, not just what. If you fixed something subtle, say what the failure
looked like — several comments in this codebase exist because the symptom was misleading.
