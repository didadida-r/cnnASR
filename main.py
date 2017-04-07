'''@file main.py
run this file to go through the neural net training procedure, look at the config files in the config directory to modify the settings'''

import sys
import os
os.system('. ./path.sh')
sys.path.append('local/kaldi')
sys.path.append('local/processing')
sys.path.append('local/neuralNetworks')
from six.moves import configparser
import tensorflow as tf
import numpy as np

import ark, prepare_data, feature_reader, batchdispenser, target_coder
import cnn

#here you can set which steps should be executed. If a step has been executed in the past the result have been saved and the step does not have to be executed again (if nothing has changed)
DNNTRAINFEATURES = True 	#required
DNNTESTFEATURES = True	 	#required if the performance of the DNN is tested
TRAIN_NNET = True			#required
TEST_NNET = True			#required if the performance of the DNN is tested

#read config file
config = configparser.ConfigParser()
config.read('conf/tfkaldi_cnn.cfg')
cnn_conf = dict(config.items('cnn'))
current_dir = os.getcwd()

# 网络输入维度
reader = ark.ArkReader(config.get('directories', 'train_features') + '/feats.scp')
_, features, _ = reader.read_next_utt()     # 这里是没有经过拼接的
input_dim = features.shape[1] * (int(cnn_conf['context_width'])*2 + 1) 
print("the input dim is:" + str(input_dim))

# 网络输出维度
numpdfs = open(config.get('directories','expdir') + '/' + config.get('cnn','gmm_name') + '/graph/num_pdfs')
num_labels = numpdfs.read()
num_labels = int(num_labels[0:len(num_labels)-1])
numpdfs.close()
print("the output labels is:" + str(num_labels))

# 后台读取batch的线程数
batch_reader_nj = 8
# 特征目录
featdir = config.get('directories','train_features')

if TRAIN_NNET:
    # shuffle the examples on disk
    prepare_data.shuffle_examples(featdir)
      
    print('------- get alignments ----------')
    alifiles = [config.get('directories', 'expdir') + '/' + config.get('cnn', 'gmm_name') + '_ali/pdf.' + str(i+1) + '.gz' for i in range(int(config.get('label', 'num_ali_jobs')))]
    alifile = config.get('directories', 'expdir') + '/' + config.get('cnn', 'gmm_name') + '/pdf.all'
    if not os.path.isfile(alifile):
      tmp = open(alifile, 'a')
      tmp.close()
    os.system('cat %s > %s' % (' '.join(alifiles), alifile))
    
    # get maxlength
    max_input_length = 0
    total_frames = 0
    with open(featdir + "/utt2num_frames", 'r') as f:
      line = f.readline()
      while line:
        x = line.split(' ')[1]
        total_frames += int(x)
        if int(x) > max_input_length:
          max_input_length = int(x)
        line = f.readline()
    # 将maxlength写入文件    
    with open(featdir + "/maxlength", 'w') as f:
      f.write("%s"%max_input_length)
      print("the utt's maxlength is: " + str(max_input_length))
      
    # with open(featdir + '/maxlength', 'r') as fid:
        # max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(featdir + '/feats_shuffled.scp', featdir + '/cmvn.scp', 
        featdir + '/utt2spk', int(cnn_conf['context_width']), max_input_length)
    
    # create a target coder
    coder = target_coder.AlignmentCoder(lambda x, y: x, num_labels)
    
    # lda在哪里做？
    dispenser = batchdispenser.AlignmentBatchDispenser(featreader, coder, 
        int(cnn_conf['batch_size']), input_dim, alifile)

    #train the neural net
    print('------- training neural net ----------')
    #create the neural net
    cnn = cnn.Cnn(input_dim, num_labels, total_frames, cnn_conf)
    cnn.train(dispenser)


# if TEST_NNET:

    # #use the neural net to calculate posteriors for the testing set
    # print '------- computing state pseudo-likelihoods ----------'
    # savedir = config.get('directories', 'expdir') + '/' + config.get('nnet', 'name')
    # decodedir = savedir + '/decode'
    # if not os.path.isdir(decodedir):
        # os.mkdir(decodedir)

    # featdir = config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')

    # #create a feature reader
    # with open(featdir + '/maxlength', 'r') as fid:
        # max_length = int(fid.read())
    # featreader = feature_reader.FeatureReader(featdir + '/feats.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', int(config.get('nnet', 'context_width')), max_length)

    # #create an ark writer for the likelihoods
    # if os.path.isfile(decodedir + '/likelihoods.ark'):
        # os.remove(decodedir + '/likelihoods.ark')
    # writer = ark.ArkWriter(decodedir + '/feats.scp', decodedir + '/likelihoods.ark')

    # #decode with te neural net
    # nnet.decode(featreader, writer)

    # print '------- decoding testing sets ----------'
    # #copy the gmm model and some files to speaker mapping to the decoding dir
    # os.system('cp %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/final.mdl', decodedir))
    # os.system('cp -r %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/graph', decodedir))
    # os.system('cp %s %s' %(config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name') + '/utt2spk', decodedir))
    # os.system('cp %s %s' %(config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name') + '/text', decodedir))

    # #change directory to kaldi egs
    # os.chdir(config.get('directories', 'kaldi_egs'))

    # #decode using kaldi
    # os.system('%s/kaldi/decode.sh --cmd %s --nj %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % (current_dir, config.get('general', 'cmd'), config.get('general', 'num_jobs'), decodedir, decodedir, decodedir, decodedir))

    # #get results
    # os.system('grep WER %s/kaldi_decode/wer_* | utils/best_wer.sh' % decodedir)

    # #go back to working dir
    # os.chdir(current_dir)
