# Makefile

CONFIG_DIR=experiments
RESULTS_DIR_BASE=results

define RUN_EXPERIMENT
	@RESULTS_DIR=$(RESULTS_DIR_BASE)/$(1)_$(shell date +%Y%m%d_%H%M)/ && \
	mkdir -p $$RESULTS_DIR && \
	cp $(CONFIG_DIR)/$(1).toml $$RESULTS_DIR/config.toml && \
	python $(CONFIG_DIR)/$(1).py --config $(CONFIG_DIR)/$(1).toml --output $$RESULTS_DIR
endef

all: fixed_clock clock_speeds

fixed_clock:
	$(call RUN_EXPERIMENT,fixed_clock)

fixed_period:
	$(call RUN_EXPERIMENT,fixed_period)

clock_speeds:
	$(call RUN_EXPERIMENT,clock_speeds)

.PHONY: all fixed_clock clock_speeds
