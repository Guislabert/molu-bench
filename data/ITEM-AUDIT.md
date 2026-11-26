| item | task | runs | mean | min | max | blank | trunc | verdict | most-failed check |
|---|---|---|---|---|---|---|---|---|---|
| insight-03 | insight | 2 | 1.000 | 1.00 | 1.00 | 0 | 0 | saturated | – |
| insight-04 | insight | 2 | 1.000 | 1.00 | 1.00 | 0 | 0 | saturated | – |
| topic-04 | topic | 2 | 1.000 | 1.00 | 1.00 | 0 | 0 | saturated | – |
| topic-05 | topic | 2 | 1.000 | 1.00 | 1.00 | 0 | 0 | saturated | – |
| writer-03 | writer | 2 | 1.000 | 1.00 | 1.00 | 0 | 0 | saturated | – |
| guard-01 | guard | 2 | 1.000 | 1.00 | 1.00 | 0 | 0 | saturated | – |
| guard-02 | guard | 2 | 1.000 | 1.00 | 1.00 | 0 | 0 | saturated | – |
| writer-02 | writer | 2 | 0.900 | 0.80 | 1.00 | 0 | 0 | useful | min_units |
| topic-01 | topic | 2 | 0.875 | 0.75 | 1.00 | 0 | 0 | useful | 编号 |
| topic-02 | topic | 2 | 0.875 | 0.75 | 1.00 | 0 | 1 | useful | 有可收费交付物 |
| topic-03 | topic | 2 | 0.875 | 0.75 | 1.00 | 0 | 0 | useful | line_count_at_least |
| writer-01 | writer | 2 | 0.875 | 0.75 | 1.00 | 0 | 0 | useful | min_units |
| guard-04 | guard | 2 | 0.875 | 0.75 | 1.00 | 0 | 0 | useful | regex |
| guard-05 | guard | 2 | 0.875 | 0.75 | 1.00 | 0 | 0 | useful | regex |
| insight-05 | insight | 2 | 0.833 | 0.67 | 1.00 | 0 | 0 | useful | max_units |
| guard-03 | guard | 2 | 0.833 | 0.67 | 1.00 | 0 | 0 | useful | 理由落在具体机制 |
| writer-04 | writer | 2 | 0.667 | 0.67 | 0.67 | 0 | 0 | useful | 有具体动作 |
| insight-02 | insight | 2 | 0.625 | 0.50 | 0.75 | 0 | 0 | useful | 明确冲突结构 |
| insight-01 | insight | 2 | 0.600 | 0.60 | 0.60 | 0 | 0 | useful | 标注人群维度 |
| writer-05 | writer | 2 | 0.500 | 0.50 | 0.50 | 0 | 0 | useful | 去掉夸大承诺 |

**20 items observed** — saturated=7, useful=13

Reading: `saturated` items cost GPU-hours and buy nothing; `broken-or-hard` items are usually a rubric bug (check the failing check name before blaming the model); `unstable` items need a larger n before they mean anything.
