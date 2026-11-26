# Dataset card — molu-bench

## What is in this repo

| file | content | redistributable |
|---|---|---|
| `data/dev.jsonl` | 20 items, 4 tasks × 5, prompts authored for this repo | yes, Apache-2.0 |
| `harness/evalkit.py` | rubric scorer, bootstrap CI, model adapters | yes, Apache-2.0 |
| `results/` | protocol + (eventually) measured outputs | yes, subject to upstream model licences |

The dev set contains **no text from the molu corpus and no text from the 899 source articles**. Every prompt was written for this repository. That is why it can be published.

## What is deliberately not in this repo

The SFT corpus. Its shape:

| quantity | value | provenance |
|---|---|---|
| source articles | 899, published 2015-09-15 .. 2018-12-27 | public WeChat Official Accounts |
| characters | 2,130,000 | as published |
| extracted content units | 16,227 | verified against source text, 100% match rate |
| unit kinds | audience segments, need statements, relationship conflicts, evidence patterns, price/offer structures, compliance failures | derived labels |
| rights | verbatim excerpts remain with the original authors | **not redistributable** |

Three consequences, stated plainly so nobody discovers them later:

1. **The corpus cannot be released**, not even under a permissive licence — the copyright is not ours to license. The molu repo already carves `molu/*/references/` out of its Apache-2.0 licence for this reason.
2. **Training is reproducible only as a procedure**, not as a byte-identical artefact: `data/CARD.md` + `data/schema.md` + the extraction rules are public, the text is not.
3. **Weights are releasable.** Fine-tuning Qwen3-8B (Apache-2.0) on locally held data produces adapters we may publish; publishing the adapter does not republish the corpus, but it does raise a memorisation question, so the released card will include a canary/overlap check against held-out articles before any weights ship.

## Task mix in the dev set

| task | what it measures | items |
|---|---|---|
| `insight` | extract audience / progress / blocker from raw user text without giving advice | 5 |
| `topic` | produce distinguishable, audience-specific topics that do not drift to "everyone" | 5 |
| `writer` | produce copy that follows plain-communication constraints and commits to no traffic promise | 5 |
| `guard` | run risk review that separates platform-machine risk from substantive content risk, with cited positions | 5 |

Why these four and not ten: each maps to a named module in molu and to a failure mode that is observable in text without human taste. Tasks that need subjective judgement ("is this interesting") are excluded on purpose — a benchmark full of taste questions measures the annotator.

## Curation rules

- One check per doctrine rule that can be verified mechanically; no check that requires a judge to decide "quality".
- Every `none_of` list holds phrases that appear in real deployments of the failure (e.g. `涨粉`, `保证`, `三天见效`) rather than invented bad words.
- Length limits are in Chinese units (1 CJK char = 1 unit, 1 latin word = 1 unit) so tokenizer choice cannot move the score.
- Banned-word checks are case- and width-insensitive where relevant; a model that evades a rule by full-width punctuation is still failing the rule.

## Known limitations

- **20 items is a smoke test, not a benchmark.** Confidence intervals on 5 items per task are wide by construction; `results/README.md` states the gating numbers before any run.
- **Deterministic rubrics under-reward good answers that took an unexpected route.** Mitigation: publish per-check failures, not just the total, so a reviewer can see which checks are brittle.
- **Single annotator** (the author) wrote the dev set. Inter-annotator agreement is an open item for the 5,000-item set.
- **Chinese-only.** Format-level claims do not transfer to English content work without re-derivation.