#!/bin/bash
#$ -cwd -V
#$ -l h_rt=00:30:00
#$ -pe smp 12
#$ -l h_vmem=6G

# activate conda and add to library path
conda activate swd8_intro_ml
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# run the CPU script
python tensorflow_ray_train_mnist_example.py --num-workers 12 --epochs 100
