#!/bin/bash
#SBATCH --job-name=pog_job
#SBATCH --output=cluster/job_output_%j.out
#SBATCH --error=cluster/job_error_%j.err
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load CUDA
cd build
./cpu/cpu_naive
./cpu/cpu_coo
./cpu/cpu_coo_struct
./cpu/cpu_csr

./gpu/gpu_coo_add_atomic
./gpu/gpu_csr
./gpu/gpu_csr_constant_memory
./gpu/gpu_csr_stride
./gpu/gpu_naive
