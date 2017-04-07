#!/bin/bash

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

logdir=tf_exp/log
$cuda_cmd $logdir/tf-sge.log run_cnn.sh