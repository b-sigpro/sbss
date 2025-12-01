# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# options
data := derev
duration := 64000

train_path := models/nfca/unet

tag := nfca
inference_name := $(tag)_$(shell date +"%Y-%m-%d-%H-%M-%S")_$(shell tr -dc 'a-z0-9' </dev/urandom | head -c 4)
inference_command := python -m sbss.nfca.pipelines.separate batch $(train_path) $$src_path $$dst_path
inference_path := results/$(inference_name)

# constants
PRINT_VARIABLES = \
	cmd job_ops -- \
	data duration -- \
	train_path -- \
	tag inference_command inference_name inference_path

SPLITS = tr cv tt
HDF5_SPLITS = tr cv
INFERENCE_SPLITS = cv tt


# stage 0: dataset preparation
stage0_dependencies = hdf5/ .0_prepare_dataset.done

.0_prepare_dataset.done:
	python ./scripts/0_prepare_dataset.py
	touch $@


# stage1: pre-processing
stage1_dependencies = \
	$(SPLITS:%=.1_add_noise.%.done) \
	$(SPLITS:%=.1_dereverberate.%.done)

.1_add_noise.%.done: | stage0
	n_tasks=$(shell find $*/mix-clean/ -name "*.wav" | wc -l) && \
	$(cmd) cpu --n_tasks=$$n_tasks $(job_ops) jobs/1_add_noise/$*.log -- python ./scripts/1_add_noise.py $*
	touch $@

.1_dereverberate.%.done: | stage0 .1_add_noise.%.done
	n_tasks=$(shell find $*/mix/ -name "*.wav" | wc -l) && \
	$(cmd) gpu --n_tasks=$$n_tasks $(job_ops) jobs/1_dereverberate/$*.log -- python ./scripts/1_dereverberate.py $*
	touch $@


# stage2: hdf5 generation
stage2_dependencies = \
	$(HDF5_SPLITS:%=.2_make_hdf5_unsupervised.$(data)_$(duration).%.done)

.2_make_hdf5_unsupervised.$(data)_$(duration).%.done: | stage1
	$(cmd) mpi --n_procs=40 $(job_ops) jobs/2_make_hdf5_unsupervised.$(data)_$(duration)/$*.log -- \
		python ./scripts/2_make_hdf5_unsupervised.py $* --data=$(data) --duration=$(duration) --parallel
	touch $@


# stage4: inference
stage4_dependencies = \
	$(INFERENCE_SPLITS:%=$(inference_path)/.4_inference.%.done)

$(inference_path)/.4_inference.%.done: | stage3
	n_tasks=$(shell find $*/mix -name "*.wav" | wc -l) && \
	src_path=$*/derev/ && \
	dst_path=$(inference_path)/$* && \
	$(cmd) gpu $(job_ops) --n_tasks=$$n_tasks $(inference_path)/jobs/4_inference.$*.log -- \
		$(inference_command)
	touch $@


# stage5: evaluation
stage5_dependencies = \
	$(INFERENCE_SPLITS:%=$(inference_path)/.5_evaluate_sdr.%.done) \
	$(INFERENCE_SPLITS:%=$(inference_path)/.5_evaluate_pesq.%.done) \
	$(INFERENCE_SPLITS:%=$(inference_path)/.5_evaluate_stoi.%.done)

$(inference_path)/.5_evaluate_sdr.%.done: | $(inference_path)/.4_inference.%.done
	$(cmd) cpu $(job_ops) $(inference_path)/jobs/5_evaluate_sdr.$*.log -- \
		python ./scripts/5_evaluate_sdr.py $* $(inference_path)
	touch $@

$(inference_path)/.5_evaluate_stoi.%.done: | $(inference_path)/.4_inference.%.done
	$(cmd) cpu $(job_ops) $(inference_path)/jobs/5_evaluate_stoi.$*.log -- \
		python ./scripts/5_evaluate_stoi.py $* $(inference_path)
	touch $@
	
$(inference_path)/.5_evaluate_pesq.%.done: | $(inference_path)/.4_inference.%.done
	$(cmd) cpu $(job_ops) $(inference_path)/jobs/5_evaluate_pesq.$*.log -- \
		python ./scripts/5_evaluate_pesq.py $* $(inference_path)
	touch $@

scores:
	python ./scripts/summarize_scores.py $(inference_name)

# clean
clean:
	-rm -r \
		./tr \
		./cv \
		./tt \
		./hdf5/* \
		./jobs

	-rm .*.done

include ../globals.mk