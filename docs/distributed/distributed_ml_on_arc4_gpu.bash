#!/bin/bash
#$ -cwd -V
#$ -l h_rt=00:15:00
#$ -l coproc_v100=1

# activate conda and add to library path
conda activate swd8_intro_ml 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# start the efficiency log for the GPU
nvidia-smi dmon -d 10 -s um -i 0 > efficiency_log &

# run the GPU script
python tensorflow_ray_train_mnist_example.py --use-gpu True --num-workers 1 --epochs 100
# python tensorflow_ray_train_transfer_learning_example.py --use-gpu True --num-workers 1 --epochs 100
# python pytorch_lightning_mnist_example.py
# python pytorch_lightning_transfer_learning_example.py

# stop the efficiency log
kill %1
