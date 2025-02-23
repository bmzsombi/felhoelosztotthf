#!/bin/bash
# Helperscriptz for running in bathc mode
# May be modified

#SBATCH -o fractal.out
#SBATCH --job-name=mandelbrot
#SBATCH --nodes=2              # Az igényelt node-ok száma
#SBATCH --ntasks-per-node=4    # Az egyes node-okon futó processzorok száma
#SBATCH --time=01:00:00        # Maximális futási idő

srun --mpi=pmi2 ./fractal

