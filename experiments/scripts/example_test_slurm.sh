#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --job-name=rgztest

module load tensorflow/1.4.0-py27-gpu
module load opencv openmpi

# if cuda driver is not in the system path, customise and add the following paths
# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PYTHONPATH=$PYTHONPATH:/home/wu082/software/lib/python2.7/site-packages
RGZ_RCNN=/flush1/wu082/proj/rgz_rcnn
mpirun -np 1 python $RGZ_RCNN/tools/test_net.py \
                    --device gpu \
                    --device_id 0 \
                    --imdb rgz_2017_testD4 \
                    --cfg $RGZ_RCNN/experiments/cfgs/faster_rcnn_end2end.yml \
                    --network rgz_test \
                    --weights $RGZ_RCNN/data/pretrained_model/rgz/D4/VGGnet_fast_rcnn-80000 \
                    --comp
