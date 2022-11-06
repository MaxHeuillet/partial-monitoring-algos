#!/bin/bash
#SBATCH --account=def-adurand   # some account
#SBATCH --time=10:00:00        # specify time limit (D-HH:MM)
#SBATCH --cpus-per-task=50     # specify number threads

python3 ./partial_monitoring/evaluation.py
