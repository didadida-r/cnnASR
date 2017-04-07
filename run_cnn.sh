#!/bin/bash
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

feats_nj=10
stage=-4

# Config:
expdir=exp/cnn          # the exp dir
logdir=$expdir/log
gmmdir=exp/tri3                 # the gmm dir
if [ ! -d $expdir ]; then
  mkdir -p $expdir
fi



if [ $stage -le -5 ]; then
echo ============================================================================
echo "                                  Make FBank & Compute CMVN                    "
echo ============================================================================

# 这里需要将前面生成的data文件夹复制到data-fbank中，否则下面的程序会修改原来data中的数据格式为fbank
# 选择不压缩
srcdir=data-fbank   # 原始数据文件以及最终的scp
fbankdir=fbank       # 中间生成文件以及生成的特征ark文件所在目录
cp -r data $srcdir || exit 1;

for x in train dev test; do
  steps/make_fbank.sh --cmd "$train_cmd" --nj 15 --compress false data-fbank/$x exp/make_fbank/$x $fbankdir
  steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $fbankdir
done

fi

# Prepare feature
if [ $stage -le -4 ]; then
echo ============================================================================
echo "                                  Store fBank features                    "
echo ============================================================================

# store_fbank.sh 为自定义

srcdir=data-fbank   # 原始数据文件以及最终的scp
fbankdir=fbank       # 中间生成文件以及生成的特征ark文件所在目录
for x in train dev test; do
  dir=$fbankdir/$x
  local/store_fbank.sh --nj $feats_nj --delta_order 2 --cmd "$train_cmd" \
    $dir $srcdir/$x $dir/log $dir/data || exit 1
done

fi

exit

if [ $stage -le -3 ];then
echo ============================================================================
echo "             Get the lda for the nnet training                            "
echo ============================================================================
 
steps/tfkaldi/get_lda.sh --cmd "$train_cmd" data-fmllr-tri3/train data/lang exp/tri3_ali $expdir
fi

if [ $stage -le -2 ];then
echo ============================================================================
echo "             Train and decode neural network acoustic model               "
echo ============================================================================

$cuda_cmd $logdir/dnn_train.log tfcnn.sh
fi

exit

if [ $stage -le -1 ];then
echo ============================================================================
echo "                           Decode with language model                     "
echo ============================================================================
#copy the gmm model and some files to speaker mapping to the decoding dir
cp exp/tri3/final.mdl $expdir
cp -r exp/tri3/graph $expdir
for x in dev test;do
  if [ ! -d $expdir/decode_${x} ]; then 
    mkdir -p $expdir/decode_${x}
  fi
  cp data-fmllr-tri3/${x}/utt2spk $expdir/decode_${x}
  cp data-fmllr-tri3/${x}/text $expdir/decode_${x}
  cp data-fmllr-tri3/${x}/stm $expdir/decode_${x}
  cp data-fmllr-tri3/${x}/glm $expdir/decode_${x}

  #decode using kaldi
  local/kaldi/decode.sh --cmd "$train_cmd" --nj 10 $expdir/graph $expdir/decode_${x} $expdir/decode_${x} | tee $expdir/decode_${x}.log || exit 1;  
done
fi
