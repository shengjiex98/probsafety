# Makefile

# Define the configuration file and output directory
CONFIG=exp1.yaml
RESULTS_DIR=results/exp1_$(shell date +%Y%m%d_%H%M)

all: preprocess train evaluate

preprocess:
	@echo "Preprocessing data..."
	python preprocess.py --config $(CONFIG)

train: preprocess
	@echo "Training model..."
	mkdir -p $(RESULTS_DIR)
	python experiment1.py --config $(CONFIG) --output_dir $(RESULTS_DIR)

evaluate: train
	@echo "Evaluating model..."
	python evaluate.py --results_dir $(RESULTS_DIR)

clean:
	@echo "Cleaning up..."
	rm -rf results/*

.PHONY: all preprocess train evaluate clean
