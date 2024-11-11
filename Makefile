# Makefile

# Define the configuration file and output directory
SCRIPT=experiments/fixed_clock.py
CONFIG=experiments/fixed_clock.toml
RESULTS_DIR=results/fixed_clock_$(shell date +%Y%m%d_%H%M)/

all: fixed_clock

fixed_clock:
	@echo "Running experiment 1..."
	cp $(CONFIG) $(RESULTS_DIR)
	python $(SCRIPT) --config $(CONFIG) --output $(RESULTS_DIR)

# clean:
# 	@echo "Cleaning up..."
# 	rm -rf results/*

.PHONY: all fixed_clock clean
