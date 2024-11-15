# Makefile

CONFIG_DIR=experiments
RESULTS_DIR_BASE=results

define RUN_EXPERIMENT
	@RESULTS_DIR=$(RESULTS_DIR_BASE)/$(1)_$(shell date +%Y%m%d_%H%M)/ && \
	mkdir $$RESULTS_DIR && \
	cp $(CONFIG_DIR)/$(1).toml $$RESULTS_DIR/config.toml && \
	python $(CONFIG_DIR)/$(1).py --config $(CONFIG_DIR)/$(1).toml --output $$RESULTS_DIR
endef

all: fixed_clock clock_speeds

1dscan:
	$(call RUN_EXPERIMENT,1dscan)

2dscan:
	$(call RUN_EXPERIMENT,2dscan)

.PHONY: all 1dscan 2dscan
