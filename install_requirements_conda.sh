#!/bin/bash

set -e
set -u
set -o pipefail

ANACONDA_HOME=/home/ec2-user/anaconda
CONDA_ENV=tensorflow2_p36

source "$ANACONDA_HOME"/bin/activate "$CONDA_ENV"

# conda install -y -c conda-forge imbalanced-learn
# conda install -y -c anaconda "tensorflow=2.0.0"
pip install "tensorflow-addons==0.6.0"

conda deactivate
