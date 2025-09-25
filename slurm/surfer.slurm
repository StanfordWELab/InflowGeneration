#!/bin/bash
#SBATCH --job-name=stitch
#SBATCH -N 
#SBATCH -n 24
#SBATCH --time=2:00:00
#SBATCH --mem 128G
#SBATCH --mem=128G
#SBATCH --partition=serc
#SBATCH --mail-type=ALL
#SBATCH --output=stitch.out

source ~/.bash_profile
module purge
module load system
module load libpng/1.2.57
# module load openmpi/4.1.0
module load openmpi/5.0.5

STITCH_FILE="./surferDomain.in"
SOLVER_DIR="/home/groups/gorle/codes/cascade-IBM/bin"
mpiexec -- $SOLVER_DIR/surfer.exe -i $STITCH_FILE