#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = WaitTimePrediction
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################
TRAINED_MODEL = $(shell find data/processed -type f -name '*.csv')
pipeline_predict: src/models/pipeline_predict.py pipeline_train models/pipeline.pkl.gz
	$(PYTHON_INTERPRETER) src/models/pipeline_predict.py models/pipeline.pkl.gz

FINAL_DATA = $(shell find data/final -type f -name '*.csv')
pipeline_train: src/models/pipeline_train.py feature_engineering $(FINAL_DATA)
	$(PYTHON_INTERPRETER) src/models/pipeline_train.py data/final models/pipeline.pkl

PROCESSED_DATA = $(shell find data/processed -type f -name '*.csv')
feature_engineering: src/models/feature_engineering.py data_cleaning $(PROCESSED_DATA)
	$(PYTHON_INTERPRETER) src/models/feature_engineering.py data/processed data/final

INTERIM_DATA = $(shell find data/interim -type f -name '*.csv')
RAW_DATA = $(shell find data/raw -type f -name '*')
data_cleaning: src/data/data_cleaning.py $(INTERIM_DATA) $(RAW_DATA)
	$(PYTHON_INTERPRETER) src/data/data_cleaning.py data/interim data/processed

### Test python environment is setup correctly
test_environment: test_environment.py
	$(PYTHON_INTERPRETER) test_environment.py



## Install Python Dependencies
requirements: requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
