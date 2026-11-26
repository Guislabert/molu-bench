PY ?= python3

.PHONY: smoke smoke-doctrine test budget table ci clean

smoke:            ## 20-item dev set, offline stub model, no API key
	$(PY) harness/cli.py --model stub --data data/dev.jsonl --quiet

smoke-doctrine:   ## same, with the molu module prompts prepended
	$(PY) harness/cli.py --model stub --data data/dev.jsonl --prompt-mode doctrine --quiet

test:             ## scorer unit tests, no network
	$(PY) -m unittest discover -s tests -v

table:            ## regenerate results/TABLE.md from results/*.json
	$(PY) harness/report.py --out results/TABLE.md

budget:           ## recompute the GPU-hour bill
	$(PY) compute/budget.py

ci: test smoke smoke-doctrine table budget  ## exactly what .github/workflows/ci.yml runs

clean:
	rm -f results/*.json
