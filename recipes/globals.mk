# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# options
cmd := $(if $(JOB_COMMAND),$(JOB_COMMAND),aiaccel-job local)
job_ops := 

# general rules
STAGES = stage0 stage1 stage2 stage3 stage4 stage5

.PHONY: all clean status scores check_train_artifacts $(STAGES) $(STAGES:%=skip-%) help
.NOTPARALLEL: stage3

# general rules
all: $(STAGES)

%: status

status:
	@echo ================================================================
	@$(foreach var,$(PRINT_VARIABLES), \
	$(if $(filter --,$(var)), \
			echo "----------------------------------------------------------------";, \
			echo -e "\033[33m$(var)\033[0m:" '$($(var))';) \
	)
	@echo ================================================================

# stage0: dataset preparation
stage0: status .WAIT | $(stage0_dependencies)
	@echo -e "\033[1;34mStage0 finished\033[0m"

hdf5/:
	@echo -e "\033[1;31m[ERROR] The directory 'hdf5/' does not exist.\033[0m"
	@echo "Please create it manually before running this script."
	@echo "Don't forget to configure stripe settings properly!"
	@echo ""
	@echo "Hint:"
	@echo "  mkdir -p hdf5/"
	@echo "  lfs setstripe -S 1m -c -1 -i -1 hdf5/"
	@echo ""
	@exit 1

skip-stage0:
	touch $(stage0_dependencies)


# stage1: pre-processing
stage1: status .WAIT | $(stage1_dependencies)
	@echo -e "\033[1;34mStage1 finished\033[0m"

skip-stage1:
	touch $(stage1_dependencies)


# stage2: hdf5 generation
stage2: status .WAIT | $(stage2_dependencies)
	@echo -e "\033[1;34mStage2 finished\033[0m"

skip-stage2:
	touch $(stage2_dependencies)


# stage3: inference
stage3_dependencies := $(train_path)/.train.done

stage3: status .WAIT | $(stage3_dependencies)
	@echo -e "\033[1;34mStage3 finished\033[0m"

$(train_path)/.train.done: | stage2 $(train_path)/config.yaml
	@set -e; \
	set -- \
		$(train_path)/train.* \
		$(train_path)/merged_config.yaml \
		$(train_path)/checkpoints \
		$(train_path)/events* \
		$(train_path)/hparams.yaml; \
	existing=""; \
	for file in "$$@"; do [ -e "$$file" ] && existing="$$existing $$file"; done; \
	if [ -n "$$existing" ]; then \
		echo "Found existing files/directories:"; printf '  %s\n' $$existing; \
		read -r -p "Delete these? [y/N]: " ans; \
		case "$$ans" in y|Y) rm -r $$existing ;; *) echo "Aborted."; exit 1 ;; esac; \
	fi
	$(cmd) train $(job_ops) \
		--n_gpus=$(shell aiaccel-config get-value $(train_path)/config.yaml n_gpus) \
		--walltime=$(shell aiaccel-config get-value $(train_path)/config.yaml walltime) \
		$(job_ops) $(train_path)/train.log -- \
			python -m aiaccel.torch.apps.train $(train_path)/config.yaml
	touch $@

skip-stage3:
	touch $(stage3_dependencies)


# stage4: inference
stage4: status .WAIT | $(stage4_dependencies)
	@echo -e "\033[1;34mStage4 finished\033[0m"

skip-stage4:
	touch $(stage4_dependencies)


# stage5: evaluation
stage5: status .WAIT | $(stage5_dependencies) .WAIT scores
	@echo -e "\033[1;34mStage5 finished\033[0m"

skip-stage5:
	touch $(stage5_dependencies)


# help
help:
	@echo "Usage: make [TARGET]"
	@echo ""
	@echo "Available targets:"
	@echo "  all                    Run all stages sequentially"
	@echo "  status                 Print selected variables (from PRINT_VARIABLES)"
	@echo "  stageN                 Run each processing stage (N=0-5)"
	@echo "  skip-stageN            Skip stageN by touching dependencies (N=0-5)"
	@echo "  clean                  Clean generated .done files (if implemented)"
	@echo "  help                   Show this help message"
	@echo ""
	@echo "Example:"
	@echo "  make stage3 -j train_path=./models/dummy/"

.DEFAULT_GOAL := help


# global configuration
REQUIRED_MAKE_VERSION := 4.4
ifeq ($(shell expr $(MAKE_VERSION) \< $(REQUIRED_MAKE_VERSION)),1)
$(error GNU Make $(REQUIRED_MAKE_VERSION) or higher is required (found $(MAKE_VERSION)))
endif
