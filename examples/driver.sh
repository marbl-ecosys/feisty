#!/bin/bash -l

#PBS -N FEISTY_driver
#PBS -A P93300606
#PBS -l select=1:ncpus=20:mpiprocs=20:mem=360GB
#PBS -l walltime=01:00:00
#PBS -q casper
#PBS -j oe
#PBS -m abe

# Just run
# $ qsub FOSI_cesm.sh
# To put this in the queue instead of running interactively
conda activate dev-feisty
mpirun -n 20 ./FOSI_cesm.py
