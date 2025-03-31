# options
cmd ?= ../clusters/abci3.py
job_ops ?= 

data ?= derev
duration ?= 64000

SPLITS = tr cv tt

train_path ?= models/nfca/unet/

# stage definitions
STAGES = stage0 stage1 stage2 stage3

.PHONY: all clean $(STAGES)

all: $(STAGES)


# general rules
.stage%.done:
	touch $@


# stage0
stage0: .stage0.done

.stage0.done: \
	.prepare_dataset.done

.prepare_dataset.done: \
	python ./scripts/prepare_dataset.py
	touch $@

# stage1
stage1: .stage1.done

.stage1.done: \
	$(SPLITS:%=.add_noise.%.done) \
	$(SPLITS:%=.dereverberate.%.done)

.add_noise.%.done: .stage0.done
	n_tasks=$(shell find $*/mix-clean -name "*.wav" | wc -l) && \
	$(cmd) cpu $(job_ops) --n_tasks=$$n_tasks jobs/add_noise/$*.log -- python ./scripts/add_noise.py $*
	touch $@

.dereverberate.%.done: .stage0.done .dereverberate.%.done
	n_tasks=$(shell find $*/mix -name "*.wav" | wc -l) && \
	$(cmd) gpu $(job_ops) --n_tasks=$$n_tasks jobs/dereverberate/$*.log -- python ./scripts/dereverberate.py $*
	touch $@


# stage2
stage2: .stage2.$(data)_$(duration).done

HDF5_SPLITS = tr cv
.stage2.$(data)_$(duration).done: \
	$(HDF5_SPLITS:%=.make_hdf5_unsupervised.$(data)_$(duration).%.done)

.make_hdf5_unsupervised.$(data)_$(duration).%.done: .stage1.done
	$(cmd) mpi $(job_ops) --n_procs=40 jobs/make_hdf5_unsupervised.$(data)_$(duration)/$*.log -- \
		python ./scripts/make_hdf5_unsupervised.py $* --data=$(data) --duration=$(duration)
	touch $@


# stage3
stage3: $(train_path)/train.done

$(train_path)/train.done: $(train_path)/config.yaml
	$(cmd) train $(job_ops) --n_gpus=8 $(train_path)/train.log -- \
		python -m aiaccel.torch.apps.train $(train_path)/config.yaml
	touch $@


# clean
clean:
	-rm -r \
		./tr \
		./cv \
		./tt \
		./hdf5/* \
		./jobs/

	-rm .*.done