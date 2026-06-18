#!/bin/bash
#SBATCH -p gpu
#SBATCH -w L402
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH -o resultado_video.%j.out
#SBATCH -e error_video.%j.err

# Cargar el entorno
module load cuda/12.2
module load Anaconda3/2024.06-1
conda activate env_sam3

# Movernos a la carpeta del proyecto
cd /Home/practicas/2026-1/SAM3/sam3

# Ejecutar el script de video
python prueba_video.py