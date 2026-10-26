.PHONY: all ingest clean eda features train benchmark aggregate dashboard smoke-test

SAMPLE_FRAC ?= 1.0
FORCE ?=

all: ingest clean eda features train benchmark aggregate

ingest:
	python src/ingest.py $(FORCE) --sample-frac $(SAMPLE_FRAC)

clean:
	python src/clean.py $(FORCE)

eda:
	python src/eda.py $(FORCE)

features:
	python src/features.py $(FORCE)

train:
	python src/train.py $(FORCE)

benchmark:
	python src/benchmark.py $(FORCE)

aggregate:
	python src/aggregate.py $(FORCE)

dashboard:
	streamlit run app/dashboard.py

smoke-test:
	python src/ingest.py --force --sample-frac 0.05
	python src/clean.py --force
	python src/eda.py --force
	python src/features.py --force
	python src/train.py --force
	python src/benchmark.py --force
	python src/aggregate.py --force
