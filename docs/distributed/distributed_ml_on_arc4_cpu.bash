#!/bin/bash
#$ -cwd
#$ -l h_rt=00:30:00
#$ -pe smp 12
#$ -l h_vmem=6G

conda activate swd8_intro_ml
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib  # (sometimes needed)

python tensorflow_ray_train_mnist_example.py --num-workers 12 --epochs 100
# python pytorch_ray_train_fashion_mnist_example.py --num-workers 12 --epochs 100
