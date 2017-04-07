!/bin/bash

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

gpu_num=1
config=conf/tfkaldi_cnn.cfg

aviliable_gpu_ids=`getgpu.sh $gpu_num` || { echo $aviliable_gpu_ids; exit 1; }
echo "Allocate GPU = $aviliable_gpu_ids."
CUDA_VISIBLE_DEVICES=$aviliable_gpu_ids python3 mnist_up/fully_connected_feed.py $config || exit 1 # run_cnn.py
