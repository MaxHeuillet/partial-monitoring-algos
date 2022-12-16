#!/bin/bash
#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=300M

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge

#module load gurobi/StdEnv2020
module load nixpkgs/16.09
#gcc/7.3.0 ipopt/3.12.13

module load python/3.7
module load scipy-stack

module load StdEnv/2016
module load gurobi/9.0.1

source /home/mheuill/projects/def-adurand/mheuill/MYENV2/bin/activate

#virtualenv --no-download ./env${VAR}
#source ./env${VAR}/bin/activate
#pip install --no-index scikit-learn
#cd $EBROOTGUROBI
#python setup.py build --build-base /tmp/${USER} install

echo "Threads 1" > gurobi.env   # set number of threads

#echo "$PWD"
#cd ~/projects/def-adurand/mheuill/attack-detection
#echo "$PWD"

python3 ./partial_monitoring/deployment_error_estimation.py 
