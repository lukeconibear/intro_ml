#!/bin/bash
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l coproc_v100=1

conda activate intro_ml 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib  # (sometimes required)

python tensorflow_ray_train_mnist_example.py --use-gpu=True --epochs 100
# python pytorch_ray_train_fashion_mnist_example.py --use-gpu=True --epochs 100
