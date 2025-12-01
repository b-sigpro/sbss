# Spatialized WSJ0-2mix

## Overview
This project automates a multi-stage audio data processing and training pipeline using `make`. Each stage performs a specific task such as data preparation, preprocessing, training, inference, and evaluation.

## Makefile Structure

### Variables
- `cmd`: Command to execute jobs (default: `../clusters/abci3.py`)
- `job_ops`: Optional job execution options
- `SPLITS`: Dataset splits (`tr`, `cv`, `tt`)
- `data`: Dataset name (default: `derev`)
- `duration`: Audio duration in samples (default: `64000`)
- `train_path`: Path to store training models
- `tag`: Tag for training (default: `nfca`)
- `inference_name`: Auto-generated name using current datetime
- `inference_path`: Path to store inference results
- `inference_command`: Python command for inference

### Constants
- `STAGES`: Defined pipeline stages
- `HDF5_SPLITS`: Splits used for HDF5 generation (`tr`, `cv`)
- `INFERENCE_SPLITS`: Splits used during inference (`cv`, `tt`)

## Pipeline Stages

### Stage 0: Dataset Preparation
```sh
make stage0
```
Runs the dataset preparation script:
- `scripts/prepare_dataset.py`

### Stage 1: Pre-processing
```sh
make stage1
```
Performs noise addition and dereverberation on all splits (`tr`, `cv`, `tt`).
- `scripts/add_noise.py`
- `scripts/dereverberate.py`

### Stage 2: HDF5 Generation
```sh
make stage2
```
Converts the preprocessed audio data into HDF5 format for training.
- `scripts/make_hdf5_unsupervised.py`

### Stage 3: Model Training
```sh
make stage3
```
Trains a model using the generated HDF5 data.
- Uses `aiaccel.torch.apps.train` with config file at `train_path/config.yaml`

### Stage 4: Inference
```sh
make stage4
```
Runs the separation model on `cv` and `tt` splits and stores the output.
- Executes: `python -m sbss.nfca.iter.separate`

### Stage 5: Evaluation
```sh
make stage5
```
Evaluates the inference results using SDR metrics.
- `scripts/evaluate_sdrs.py`

## Clean Up
```sh
make clean
```
Removes generated data, jobs, and `.done` flags.

## Running the Full Pipeline
To execute the entire pipeline:
```sh
make all
```
This will run all stages from data preparation to evaluation.

## Notes
- Each stage creates a `.done` file to avoid redundant execution.
- Jobs are submitted via the command specified in `cmd` (default is a Python script for job submission on ABCI cluster).

