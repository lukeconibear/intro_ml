#!/bin/bash -l
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l coproc_v100=1

# ----------

# 1. activate the appropriate conda environment
conda activate tf_ray_arc4
# conda activate pytorch_ray_arc4

# 2. append the path for the conda environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# ----------

# 3. run the python script

python tensorflow_mnist_example.py
# python tensorflow_ray_mnist_example.py --use-gpu True --num-workers 1
# python pytorch_ray_train_fashion_mnist_example.py --use-gpu True --num-workers 1


