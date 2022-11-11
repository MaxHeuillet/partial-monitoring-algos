#!/bin/bash

module --force purge

module load nixpkgs/16.09
#gcc/7.3.0 ipopt/3.12.13

module load python/3.7
module load scipy-stack

module load StdEnv/2016
module load gurobi/9.0.1


virtualenv --no-download ./env${VAR}

pip install --no-index scikit-learn

cd $EBROOTGUROBI
python setup.py build --build-base /tmp/${USER} install

