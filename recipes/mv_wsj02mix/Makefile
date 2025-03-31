# options
cmd = ../clusters/abci3.py
job_ops ?= 

SPLITS = tr cv tt
FSD50K_SPLITS = dev eval

data ?= mix
duration ?= 64000

train_path ?= models/tv_nfca/unet/


# stage definitions
STAGES = stage0 stage1 stage2

.PHONY: all clean $(STAGES)

all: $(STAGES)


# general rules
.%.done:
	python scripts/$(*).py

.stage%.done:
	touch $@


# stage0
stage0: .stage0.done

.stage0.done: \
	.prepare_dataset.done \
	.gather_wsj0_info.done \
	$(SPLITS:%=.generate_conditions.%.done) \
	$(SPLITS:%=.generate_mixtures.%.done)

.prepare_dataset.done:
	python ./scripts/prepare_dataset.py
	touch $@

.gather_wsj0_info.done:
	./scripts/gather_wsj0_info.py
	touch $@

.generate_conditions.%.done: .gather_wsj0_info.done .prepare_dataset.done
	$(cmd) mpi jobs/generate_conditions/$*.log -- python ./scripts/generate_conditions.py $*

.generate_mixtures.%.done: .generate_conditions.%.done
	n_tasks=$(shell python -c 'from aiaccel.utils import load_config; print(load_config("config.yaml").n_mix_dict["$*"])') && \
	$(cmd) gpu jobs/generate_mixtures/$*.log -- python ./scripts/generate_mixtures.py $*

# stage1
stage1:


# stage2
stage2: .stage2.$(data)_$(duration).done

HDF5_SPLITS = tr cv
.stage2.$(data)_$(duration).done: \
	$(HDF5_SPLITS:%=.make_hdf5_unsupervised.$(data)_$(duration).%.done)

.make_hdf5_unsupervised.$(data)_$(duration).%.done: .stage1.done
	$(cmd) mpi --n_procs=48 jobs/make_hdf5_unsupervised/$*.log -- python ./scripts/make_hdf5_unsupervised.py $* --parallel


# stage3
stage3: $(train_path)/train.done

$(train_path)/train.done: $(train_path)/config.yaml
	$(cmd) train $(job_ops) --n_gpus=8 $(train_path)/train.log -- \
		python -m aiaccel.torch.apps.train $(train_path)/config.yaml


# clean
clean:
	-rm -r \
		./wsj0 \
		./tr \
		./cv \
		./tt \
		./traj_plot \
		./hdf5/* \
		./jobs/

	-rm .*.done