.PHONY: setup selftest test discover download all quality
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .
selftest:      ## full pipeline on SYNTHETIC data (no network/credentials)
	python -m eplanes selftest
test:
	python -m pytest -q
discover:
	python -m eplanes discover
download:
	python -m eplanes download
all:           ## everything after download -> outputs/eplanes_results_*.zip
	python -m eplanes all
