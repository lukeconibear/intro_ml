#!/bin/bash
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l coproc_v100=1

conda activate swd8_intro_ml 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib  # (sometimes required)

python tensorflow_ray_train_mnist_example.py --use-gpu True --num-workers 2 --epochs 100
# python pytorch_ray_train_fashion_mnist_example.py --use-gpu True --num-workers 2 --epochs 100
