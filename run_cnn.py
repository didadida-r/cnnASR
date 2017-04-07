#coding : utf-8
import sys
import os
os.system('. ./path.sh')
sys.path.append('local/kaldi')
sys.path.append('local/processing')
sys.path.append('local/features')
sys.path.append('local/neuralNetworks')

from six.moves import configparser
import kaldiInterface
import prepare_data
import kaldi_reader
import ark
import tensorflow as tf
import nnet

TRAIN_NNET = True
TEST_NNET = True

usage = '''
usage:
  run_dnn.py <net_config_file>
'''
if len(sys.argv) != 2:
  print (usage)
  sys.exit(1)
net_config_file = sys.argv[1]

# 查看配置
print('read config file')
#read config file
config = configparser.ConfigParser()
config.read(net_config_file)
current_dir = os.getcwd()

logdir = config.get('directories','expdir') + '/' + config.get('nnet','name')
os.system('mkdir -p %s'%logdir)

# 特征向量维度
print('get the feature input dim')
#get the feature input dim
reader = ark.ArkReader(config.get('directories','train_features') + '/feats.scp')
(_,features,_) = reader.read_next_utt()
input_dim = features.shape[1]
print('feature dim is : ' + str(input_dim))

# 输出维度
print('get number of output labels')
#get number of output labels
numpdfs = open(config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/graph/num_pdfs')
num_labels = numpdfs.read()
num_labels = int(num_labels[0:len(num_labels)-1])
numpdfs.close()

#lda_dim = int(os.popen("cat %s/lda_dim"%(config.get('directories','expdir') + '/' + config.get('nnet','name'))).read())
#print('the input dim' + repr(input_dim) + '  the lda dim ' + repr(lda_dim) + ' the num labels ' + repr(num_labels))

#create the neural net   
#Nnet = nnet.Nnet(config, input_dim, num_labels, True, config.get('directories','expdir') + '/' + config.get('nnet','name') + '/lda.mat', lda_dim)

#create the neural net without lda
Nnet = nnet.Nnet(config, input_dim, num_labels)

print('start training')
if TRAIN_NNET:
  #only shuffle if we start with initialisation
  if config.get('nnet','starting_step') == '0':
    # 打乱数据
    #shuffle the examples on disk
    print('------- shuffling examples ----------')
    featfile = config.get('directories','train_features')
    prepare_data.shuffle_examples(featfile,featfile)
  
  #put all the alignments in one file
  gmmfiles = config.get('directories','expdir') + '/' + config.get('nnet','gmm_name')
  alifiles = [ gmmfiles + '_ali/ali.' + str(i+1) + '.gz' for i in range(int(config.get('label','num_ali_jobs')))]
  alifilebinary = config.get('directories','expdir') + '/' + config.get('nnet','name') + '/ali.binary.gz'
  alifile = config.get('directories','expdir') + '/' + config.get('nnet','name') + '/ali.text.gz'
  os.system('cat %s > %s' % (' '.join(alifiles), alifilebinary))
  os.system('copy-int-vector ark:"gunzip -c %s |" ark:- | ali-to-pdf %s_ali/final.alimdl ark:- ark,t:- | gzip -c > %s'%(alifilebinary,gmmfiles,alifile))
  
  #train the neural net
  # 这里将特征和对齐结果放入网络中
  print('------- training neural net ----------')
  Nnet.train(config.get('directories','train_features'), False, alifile)
    

if TEST_NNET:
  #use the neural net to calculate posteriors for the testing set
  print('------- computing state pseudo-likelihoods ----------')
  savedir = config.get('directories','expdir') + '/' + config.get('nnet','name')
  
  decodedir = savedir + '/decode_dev'
  if not os.path.isdir(decodedir):
    os.mkdir(decodedir)
  Nnet.decode(config.get('directories','dev_features'), False, decodedir)
  
  decodedir = savedir + '/decode_test'
  if not os.path.isdir(decodedir):
    os.mkdir(decodedir)
  Nnet.decode(config.get('directories','test_features'), False, decodedir)
print ('Decode with acoustic model finish.')
