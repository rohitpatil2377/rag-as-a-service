# Human-labelled calibration golden

The synthetic (G8) golden is fine for a smoke test but too weak to CALIBRATE thresholds — its
questions retrieve poorly and it has no faithfulness negatives, so a sweep on it produces
artefacts (measured: it "recommended" abstain 0.02, an artefact of junk positives). A real
calibration needs a small **human-labelled** set. This is that template.

## What to write (~40 rows)

Fill `sample-data/golden/golden-template.json` (copy it per KB). Three `kind`s, each feeds a
gate population:

| `kind` | What it is | Feeds |
|---|---|---|
| **answerable** | a question the corpus genuinely answers | abstain POSITIVES (must pass) + faithfulness POSITIVES (the real answer should score supported) |
| **unanswerable** | plausible-sounding but NOT in the corpus | abstain NEGATIVES (the gate must stop these) |
| **unfaithful** | a plausible **wrong** answer the sources don't support | faithfulness NEGATIVES |

Aim roughly for **20 answerable · 12 unanswerable · 8 unfaithful**. Balance matters more than
volume — 40 good rows beat 400 auto-generated ones.

### Writing guidance
- **answerable:** real questions a user would ask, that the docs actually answer. Put the key
  fact in `expected_answer`. Vary difficulty (some obvious, some needing a specific passage).
- **unanswerable:** *off-topic but realistic* — not gibberish. "What is our GDPR data-residency
  clause?" against an HR-leave corpus is a good negative; "asdf qwerty" is not. These are what
  the abstain gate must catch.
- **unfaithful:** take an answerable question and write an answer that is fluent but **wrong /
  unsupported** — a contradicted number, an invented policy, a hallucinated entity. `answer_to_judge`
  is that bad answer. These need `--judge-url` (an external judge scores your supplied answer
  against the retrieved sources); they are skipped without one.

## Running it

```bash
# Abstain calibration (retrieve-only, robust, no judge needed):
uv run python scripts/calibrate.py --kb <kb> --golden your-golden.json

# + faithfulness calibration (needs an external judge to score answers, incl. the unfaithful ones):
uv run python scripts/calibrate.py --kb <kb> --golden your-golden.json \
  --judge-url http://judge:8000/v1 --judge-model <strong-model> --judge-key "$KEY" --report out.md
```

The harness sweeps each gate's two populations and reports the current threshold vs the
Youden-optimal split (evidence only — a human sets the number, per ADR-0005/0015). With real
answerable/unanswerable/unfaithful rows you get a **trustworthy** optimum, not an artefact.

## Why not just auto-generate more?

Measured, on this stack: a 4-row hand golden cleanly validated the live abstain 0.5 (gap 0.135);
a 30-row *synthetic* golden produced a noisier, misleading result (half the "answerable" rows
retrieved ~0). More auto-generated data made calibration **worse**. Golden quality, not size, is
the bottleneck — hence this human template.
