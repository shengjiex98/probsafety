# Makefile

# Define the configuration file and output directory
SCRIPT=experiments/exp1.py
CONFIG=experiments/exp1.toml
RESULTS_DIR=results/exp1_$(shell date +%Y%m%d_%H%M)

all: experiment1

experiment1:
	@echo "Running experiment 1..."
	python $(SCRIPT) --config $(CONFIG) --output $(RESULTS_DIR)

# clean:
# 	@echo "Cleaning up..."
# 	rm -rf results/*

.PHONY: all experiment1 clean
