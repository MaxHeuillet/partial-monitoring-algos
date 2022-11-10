#!/bin/bash
#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --time=01:00:00

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL



module --force purge

module load nixpkgs/16.09 
#gcc/7.3.0 ipopt/3.12.13

module load python/3.7 
module load scipy-stack

module load StdEnv/2016
module load gurobi/9.0.1


source /home/mheuill/projects/def-adurand/mheuill/MYENV2/bin/activate

#pip install scikit-learn --no-index

echo "Threads ${SLURM_CPUS_ON_NODE:-1}" > gurobi.env   # set number of threads

python3 ./partial_monitoring/experiment.py
