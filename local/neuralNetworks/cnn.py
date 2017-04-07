'''@file cnn.py
contains the functionality for a Kaldi style neural network'''

import os
os.system('. ./path.sh')
import tensorflow as tf
import numpy as np

class Cnn(object):   

    def __init__(self, input_dim, num_labels, total_frames, cnn_conf_dict):      
        self.num_labels = num_labels    # 网络输出维度
        self.input_dim = input_dim      # 网络输入维度
        self.total_frames = total_frames# 所有帧的个数
        self.left_frames = total_frames
        
        # 初始学习率
        self.init_learning_rate = float(cnn_conf_dict['init_learning_rate'])
        # decay_rate
        self.decay_rate = float(cnn_conf_dict['decay_rate'])
        # batch大小，单位为帧数
        self.minibatch_size = int(cnn_conf_dict['minibatch_size'])
        # self.dropout, probability to keep units
        self.dropout = float(cnn_conf_dict['dropout'])
        # 遍历所有数据的次数
        self.num_epoch = int(cnn_conf_dict['num_epoch']) 
        # 打印轮次
        self.display_step = int(cnn_conf_dict['display_step'])
        # The log dir
        self.TensorboardDir = 'summary/log'
   
        self.weights = {
            'conv1': self.weight_with_loss('conv1_weights', [7, 1, 1, 128], None),
            'conv2': self.weight_with_loss('conv2_weights', [4, 1, 128, 256], None),
            'full1': self.weight_with_loss('full1_weights', [18*1*256, 1024], None), # 9 equals 36/(pool_size*pool_times)
            'out': self.weight_with_loss('out_weights', [1024, self.num_labels], None),
        }

        self.biases = {
            'conv1': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='conv1_biases'),
            'conv2': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='conv2_biases'),
            'full1': tf.Variable(tf.constant(0.1, shape=[1024], dtype=tf.float32), trainable=True, name='full1_biases'),
            'out': tf.Variable(tf.constant(0.1, shape=[self.num_labels], dtype=tf.float32), trainable=True, name='out_biases'),
        } 
    
    # define weight with normalization loss     
    def weight_with_loss(self, name, shape, factor):
        var = tf.get_variable(name, shape=shape, dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        if factor is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), factor, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return var
            
    # conv for the first layer
    def conv2d_FL(self, x, W, b, strides=1):
        self.variable_summary(W, W.name.split(':')[0])
        self.variable_summary(b, b.name.split(':')[0])

        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        # from shape: [N, 36, 27, 32] --> shape: [N, 36, 1, 32]
        x = tf.reduce_sum(x, 2, keep_dims=True)     # 将卷积结果按照tensor的第二个维度求和
        return tf.nn.relu(x) 

    # conv for the other layers    
    def conv2d(self, x, W, b, strides=1):
        self.variable_summary(W, W.name.split(':')[0])
        self.variable_summary(b, b.name.split(':')[0])

        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                                padding='SAME')
    def fl_net(self, x, W, b):  
        self.variable_summary(W, W.name.split(':')[0])
        self.variable_summary(b, b.name.split(':')[0])

        x = tf.add(tf.matmul(x, W), b)
        return x

    # Feed-Forward Model
    def conv_net(self, x, weights, biases, dropout):
        with tf.variable_scope('Conv1'):
            # 第一层使用了特殊的卷积
            conv1 = self.conv2d_FL(x, weights['conv1'], biases['conv1'])    # shape: [N, 36, 1, 32]
            # Max Pooling (down-sampling), 这里需要添加按27求和
            conv1 = self.maxpool2d(conv1, k=2)   # shape: [N, 18, 27, 32]

        with tf.variable_scope('Conv2'):
            conv2 = self.conv2d(conv1, weights['conv2'], biases['conv2'])    # shape: [N, 18, 1, 64]
            # Max Pooling (down-sampling)
            #conv2 = self.maxpool2d(conv2, k=2)   # shape: [N, 9, 27, 64]

        with tf.variable_scope('FL1'):
            # 将con2的结果reshape成FL层的输入，其shape为[N, 9*64]
            fc1 = tf.reshape(conv2, [-1, weights['full1'].get_shape().as_list()[0]])
            fc1 = self.fl_net(fc1, weights['full1'], biases['full1'])     # shape: [N, 1024]
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.dropout)

        with tf.variable_scope('FL2'):
            out = self.fl_net(fc1, weights['out'], biases['out'])
            return out

    # Return the final loss
    # loss = weight_loss + cross_entropy_loss    
    def loss(self, logits, labels):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.add_to_collection('losses', cross_entropy_loss)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self, dispenser):
        # 定义网络输入
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, self.input_dim])
            y = tf.placeholder(tf.int32, [None])
            keep_prob = tf.placeholder(tf.float32)

            inputs = tf.reshape(x, shape=[self.minibatch_size, 36, 27, 1])
            labels = tf.one_hot(indices=y, depth=self.num_labels, on_value=1, off_value=0, axis=-1)
            tf.summary.image('inputs', inputs, max_outputs=10)

        # Construct model
        with tf.name_scope('models'):
            pred = self.conv_net(inputs, self.weights, self.biases, keep_prob)

        # Define loss and optimizer
        with tf.name_scope('loss'):
            cost = self.loss(pred, labels)

        with tf.name_scope('train_steps'):
            global_step = tf.Variable(1, dtype=tf.int32)
            # 学习率变量
            #learning_rate = tf.Variable(self.init_learning_rate, tf.float32)
            learning_rate = tf.train.exponential_decay(self.init_learning_rate, global_step, 
                            self.total_frames//self.minibatch_size, self.decay_rate, staircase=True)
            tf.summary.scalar("lr", learning_rate)

            #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
            #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
            # Evaluate model
            # argmax取的是最大值的下标
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', accuracy) 

        merged = tf.summary.merge_all()
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            total_steps = (self.total_frames//self.minibatch_size)*self.num_epoch

            # Init the log writer
            summary_writer = tf.summary.FileWriter(self.TensorboardDir, tf.get_default_graph())

            # Keep training until reach max iterations
            while step < total_steps:
                batch_x, batch_y, looped = dispenser.get_minibatch(self.minibatch_size)
                summary, lr, _ = sess.run([merged, learning_rate, optimizer], feed_dict={x: batch_x, 
                    y: batch_y,keep_prob: self.dropout})

                if step % self.display_step == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y,keep_prob: 1.})
                    print("Iter " + str(step) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc) + ", Learning Rate= " + \
                          "{:.5f}".format(lr))
                step += 1

                summary_writer.add_summary(summary, step)

                # if looped:
                #     epoch_count += 1.0
                #     learning_rate = learning_rate * tf.pow(self.decay_rate, tf.constant(epoch_count))
                #     print("new epoch: " + str(epoch_count))
                #     print("new learning_rate" + str(lr))
            print("Optimization Finished!")
            summary_writer.close()

    def variable_summary(self, var, name):
        #with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.histogram(name, var)     # Add op to record the distribution of the variable
        tf.summary.scalar('mean/' + name, mean)
        tf.summary.scalar('stddev/' + name, stddev)