#!/bin/bash

#SBATCH --job-name=RLTL2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=71:59:59
#SBATCH --partition gpu8_medium
#SBATCH --gres=gpu:1

module purge
module load condaenvs/gpu/machinelearning
cd /gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/mohit/RL-project-transfer-learning

python train.py --config configs/walker_stand_ddpg_dmcnn2.yaml