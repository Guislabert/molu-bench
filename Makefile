PY ?= python3

.PHONY: smoke budget ci clean

smoke:            ## 20-item dev set, offline stub model, no API key
	$(PY) harness/cli.py --model stub --data data/dev.jsonl --quiet

budget:           ## recompute the GPU-hour bill
	$(PY) compute/budget.py

ci: smoke budget  ## exactly what .github/workflows/ci.yml runs

clean:
	rm -f results/*.json
