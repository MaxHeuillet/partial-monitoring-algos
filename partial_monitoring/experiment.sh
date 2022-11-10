#!/bin/bash
#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
<<<<<<< HEAD
#SBATCH --cpus-per-task=60
#SBATCH --time=03:00:00
=======
#SBATCH --cpus-per-task=100
#SBATCH --time=02:00:00
>>>>>>> 4614a26dd63e419e54b2c894341b05fd158a19c6

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

python3 ./partial_monitoring/experiment2.py --horizon ${HORIZON} --n_folds ${NFOLDS} --game ${GAME} --alg ${ALG} --task ${TASK} --context_type ${CONTEXT_TYPE}
