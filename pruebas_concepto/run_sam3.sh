#!/bin/bash
#SBATCH -p gpu
#SBATCH -w L402
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH -o resultados_imagenes/resultado_sam3.%j.out
#SBATCH -e resultados_imagenes/error_sam3.%j.err

module load cuda/12.2
module load Anaconda3/2024.06-1

cd /Home/practicas/2026-1/SAM3

export HF_TOKEN=

# Ejecutar el script con el entorno conda
/Home/practicas/2026-1/.conda/envs/env_sam3/bin/python prueba_inicial_img.py