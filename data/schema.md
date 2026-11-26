# Rubric schema

One JSON object per line in `data/*.jsonl`. The scorer in `harness/evalkit.py` reads exactly these fields; unknown check kinds abort instead of scoring 0, so a typo cannot silently deflate a model.

## Item

```json
{
  "id": "insight-01",
  "task": "insight",
  "prompt": "…the text sent to the model, verbatim…",
  "checks": [ { "kind": "contains", "value": "人群", "points": 1, "label": "标注人群维度" } ],
  "tags": ["jtbd", "b2c"]
}
```

| field | required | meaning |
|---|---|---|
| `id` | yes | `{task}-{nn}`, unique |
| `task` | yes | `insight` \| `topic` \| `writer` \| `guard` |
| `prompt` | yes | the full model input, including any excerpt. Never drawn from the un-redistributable corpus. |
| `checks` | yes | non-empty list; `max_points` is their sum |
| `tags` | no | free-form, used for slicing results |

## Check

| `kind` | fields | passes when |
|---|---|---|
| `contains` | `value` | substring present |
| `not_contains` | `value` | substring absent |
| `all_of` | `value[]` | every substring present |
| `none_of` | `value[]` | no substring present |
| `regex` | `value` | pattern matches (multiline) |
| `min_units` / `max_units` | `value` | length in units is ≥ / ≤ value |
| `line_count_at_least` | `value` | non-blank lines ≥ value |

`points` defaults to 1. `label` is what appears in the per-check failure table, so it should name the doctrine rule, not the mechanic — `"无效果承诺"` beats `"none_of(保证)"`.

## Units

1 CJK character = 1 unit; 1 run of latin letters/digits = 1 unit. Chosen so that switching tokenizer or model does not move length-based scores.

## Writing a check that survives contact with a real model

- **Ban the phrase, not the sentiment.** `none_of: ["保证", "一定会"]` is auditable; `not_verbose` is not.
- **Prefer structure over content where possible.** `line_count_at_least: 5` catches a model that answers "给 5 个选题" with three, and it cannot be gamed by vocabulary.
- **One rule, one check.** Two rules in one regex means a partial failure scores zero and hides which rule broke.
- **No check that a judge should decide.** If the pass condition is "this is good writing", it belongs in the qualitative review, not the rubric.