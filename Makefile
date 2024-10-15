# Makefile
PACKAGE_DIR = src/probsafety
INSTALL_TIMESTAMP = .install-timestamp

# Define the configuration file and output directory
SCRIPT=experiments/exp1.py
CONFIG=experiments/exp1.toml
RESULTS_DIR=results/exp1_$(shell date +%Y%m%d_%H%M)

all: install_package experiment1

install_package: $(INSTALL_TIMESTAMP)

$(INSTALL_TIMESTAMP): $(shell find $(PACKAGE_DIR) -type f)
	@echo "Installing package in editable mode..."
	pip install -e .
	@touch $(INSTALL_TIMESTAMP)

experiment1:
	@echo "Running experiment 1..."
	python $(SCRIPT) --config $(CONFIG) --output $(RESULTS_DIR)

# clean:
# 	@echo "Cleaning up..."
# 	rm -rf results/*

.PHONY: all install_package experiment1 clean
