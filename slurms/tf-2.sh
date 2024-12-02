#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -J tf-2
#SBATCH -o /hpc/home/yc583/Biostat-823/Final_Project/slurms/outputs/tf-2.out
#SBATCH -e /hpc/home/yc583/Biostat-823/Final_Project/slurms/outputs/tf-2.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p scavenger-gpu
#SBATCH --nice=100
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#

hostname
nvidia-smi
python /hpc/home/yc583/Biostat-823/Final_Project/Deepstarr-train.py /hpc/home/yc583/Biostat-823/Final_Project/config/config-tf-2.json /hpc/home/yc583/Biostat-823/Final_Project/data/Sequences_activity_all.txt /hpc/home/yc583/Biostat-823/Final_Project/saved_models tf-2