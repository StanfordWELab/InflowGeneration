#!/bin/bash
#SBATCH --job-name=charles
#SBATCH -N 4
#SBATCH -n 128
#SBATCH --time=72:00:00
#SBATCH --mem 128G
#SBATCH --partition=serc
#SBATCH --mail-type=ALL
#SBATCH --output=charles.out

export CTI_LICENSE=$HOME/Cascade/license.dat

source ~/.bash_profile
module purge
module load system
module load libpng/1.2.57
# module load openmpi/4.1.0
module load openmpi/5.0.5
SOLVER_DIR="/home/groups/gorle/codes/cascade-IBM/bin"

mpiexec -- $SOLVER_DIR/charles_helm.exe -i "./start.in" #> charles.out
echo "Starter simulation complete"
mpiexec -- $SOLVER_DIR/charles_helm.exe -i "./charles.in" #> charles.out