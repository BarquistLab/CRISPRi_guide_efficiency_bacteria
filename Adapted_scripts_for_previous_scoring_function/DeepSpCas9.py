import os
from os.path import exists
import numpy as np
from operator import itemgetter
# import pyexcel as pe
from random import shuffle
import sys
import tensorflow as tf
from os import system
import scipy.misc
import scipy.stats
from scipy.stats import rankdata
from random import shuffle

np.set_printoptions(threshold=sys.maxsize)

##############################################################################
##############################################################################
#READ ME
#READ ME
#READ ME
##############################################################################
# Final result is TEST_Z in Model_Finaltest
# TEST_Z will be saved in RANK_final_{}.txt".format(best_model_path.split('/')[1]
# ex) best_model_path_list = ["./best/"] => will be saved in RANK_final_best.txt
##############################################################################
##############################################################################
path = '/dataset/'
parameters = {}
#parameters['0'] = [1, Seq Column, -1, 1, 'File name.xlsx', 'Sheet name']
# ex) parameters['0'] = [1, 0, -1, 1, 'additionaltest.xlsx', 'Sheet1']
parameters['0'] = ['sample.txt']
        
#Parameters
TEST_NUM_SET = [0]
#best_model_path_list = ["Best model saved directory"]
# ex) best_model_path_list = ["./6_11_2_9_best/"]
best_model_path_list = ["PATH_TO_THE_WEIGHT/DeepSpCas9_weight/"]

# Model
length = 30

class Seq_deepCpf1(object):
    def __init__(self, filter_size, filter_num, node_1 = 80, node_2 = 40, l_rate = 0.005):
        self.inputs = tf.placeholder(tf.float32, [None, 1, length, 4])
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.is_training = tf.placeholder(tf.bool)
        def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
            # setup the filter input shape for tf.nn.conv_2d
            conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                              num_filters]

            # initialise weights and bias for the filter
            weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                              name=name+'_W')
            bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

            # setup the convolutional layer operation
            out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='VALID')

            # add the bias
            out_layer += bias

            # apply a ReLU non-linear activation
            out_layer = tf.layers.dropout(tf.nn.relu(out_layer), 0.3, self.is_training)

            # now perform max pooling
            ksize = [1, pool_shape[0], pool_shape[1], 1]
            strides = [1, 1, 2, 1]
            out_layer = tf.nn.avg_pool(out_layer, ksize=ksize, strides=strides, 
                                       padding='SAME')
            return out_layer

        L_pool_0 = create_new_conv_layer(self.inputs, 4, filter_num[0], [1, filter_size[0]], [1, 2], name='conv1')
        L_pool_1 = create_new_conv_layer(self.inputs, 4, filter_num[1], [1, filter_size[1]], [1, 2], name='conv2')
        L_pool_2 = create_new_conv_layer(self.inputs, 4, filter_num[2], [1, filter_size[2]], [1, 2], name='conv3')
        with tf.variable_scope('Fully_Connected_Layer1'):
            layer_node_0 = int((length-filter_size[0])/2)+1
            node_num_0 = layer_node_0*filter_num[0]
            layer_node_1 = int((length-filter_size[1])/2)+1
            node_num_1 = layer_node_1*filter_num[1]
            layer_node_2 = int((length-filter_size[2])/2)+1
            node_num_2 = layer_node_2*filter_num[2]
            L_flatten_0 = tf.reshape(L_pool_0, [-1, node_num_0])
            L_flatten_1 = tf.reshape(L_pool_1, [-1, node_num_1])
            L_flatten_2 = tf.reshape(L_pool_2, [-1, node_num_2])
            L_flatten = tf.concat([L_flatten_0, L_flatten_1, L_flatten_2], 1, name='concat')
            node_num = node_num_0 + node_num_1 + node_num_2
            W_fcl1 = tf.get_variable("W_fcl1", shape=[node_num, node_1])
            B_fcl1 = tf.get_variable("B_fcl1", shape=[node_1])
            L_fcl1_pre = tf.nn.bias_add(tf.matmul(L_flatten, W_fcl1), B_fcl1)
            L_fcl1 = tf.nn.relu(L_fcl1_pre)
            L_fcl1_drop = tf.layers.dropout(L_fcl1, 0.3, self.is_training)

        with tf.variable_scope('Fully_Connected_Layer2'):
            W_fcl2 = tf.get_variable("W_fcl2", shape=[node_1, node_2])
            B_fcl2 = tf.get_variable("B_fcl2", shape=[node_2])
            L_fcl2_pre = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_fcl2), B_fcl2)
            L_fcl2 = tf.nn.relu(L_fcl2_pre)
            L_fcl2_drop = tf.layers.dropout(L_fcl2, 0.3, self.is_training)
            
        with tf.variable_scope('Output_Layer'):
            W_out = tf.get_variable("W_out", shape=[node_2, 1])#, initializer=tf.contrib.layers.xavier_initializer())
            B_out = tf.get_variable("B_out", shape=[1])#, initializer=tf.contrib.layers.xavier_initializer())
            self.outputs = tf.nn.bias_add(tf.matmul(L_fcl2_drop, W_out), B_out)

        # Define loss function and optimizer
        self.obj_loss =  tf.reduce_mean(tf.square(self.targets - self.outputs))
        self.optimizer = tf.train.AdamOptimizer(l_rate).minimize(self.obj_loss)  

def Model_Finaltest(sess, TEST_X, filter_size, filter_num, if3d, model, args, load_episode, best_model_path):
    test_batch = 500
    test_spearman = 0.0
    optimizer = model.optimizer
    TEST_Z = np.zeros((TEST_X.shape[0], 1), dtype=float)
    file=open('FILE_TO_SAVE_SCORE','w')
    for i in range(int(np.ceil(float(TEST_X.shape[0])/float(test_batch)))):
        Dict = {model.inputs: TEST_X[i*test_batch:(i+1)*test_batch], model.is_training: False}
        TEST_Z[i*test_batch:(i+1)*test_batch] = sess.run([model.outputs], feed_dict=Dict)[0]
    for score in np.array(TEST_Z.reshape([np.shape(TEST_Z)[0]])):
        file.writelines(str(score)+'\n')
    file.close()
    return

def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length = len(data[0])
    DATA_X = np.zeros((len(data),1,length,4), dtype=int)
    print(np.shape(data), len(data), length)
    for l in range(len(data)):
        for i in range(length):
            try: data[l][i]
            except: print(data[l], i, length, len(data))
            if data[l][i]in "Aa":
                DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                DATA_X[l, 0, i, 3] = 1
            else:
                print("Non-ATGC character " + data[l], data[l][i])
                sys.exit()
    print("Preprocessing the sequence done")
    return DATA_X
    
def getseq(filenum):
#    param = parameters['%s'%filenum]
#    FILE = open(path+param[0], "r")
    
    
#    data = FILE.readlines()
#    data_n = len(data) - 1
    seq = []
    #CA = np.zeros((data_n, 1), dtype=int)
#    for l in range(1, data_n+1):
#        data_split = data[l].split()
#        seq.append(data_split[1])
        #CA[l-1,0] = int(data_split[2])*100
#    FILE.close()
    import pandas
    df1=pandas.read_csv('PATH_TO_gRNA_TABLE.csv',sep='\t')
    for i in df1.index:
        seq.append(df1['sequence_30nt'][i])
    processed_full_seq = preprocess_seq(seq)
    return processed_full_seq, seq
    
#TensorFlow config
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

best_model_cv = 0.0
best_model_list = []
for best_model_path in best_model_path_list:
    for modelname in os.listdir(best_model_path):
        if "meta" in modelname:
            best_model_list.append(modelname[:-5])
TEST_X = []
TEST_X_nohot = []
for TEST_NUM in TEST_NUM_SET:
    tmp_X, tmp_X_nohot = getseq(TEST_NUM)
    TEST_X.append(tmp_X)
    TEST_X_nohot.append(tmp_X_nohot)

for index in range(len(best_model_list)):
    best_model_path = best_model_path_list[index]
    best_model = best_model_list[index]
    print(best_model_path + best_model)
    valuelist = best_model.split('-')
    fulllist = []
    
    for value in valuelist:
        if value == 'True':
            value=True
        elif value == 'False':
            value=False
        else:
            try:
                value=int(value)
            except:
                try:value=float(value)
                except: pass
        fulllist.append(value)
    
    if fulllist[2:][-3] is True:
        if3d, filter_size_1, filter_size_2, filter_size_3, filter_num_1, filter_num_2, filter_num_3, l_rate, load_episode, inception, node_1, node_2 = fulllist[2:]
        filter_size = [filter_size_1, filter_size_2, filter_size_3]
        filter_num  = [filter_num_1, filter_num_2, filter_num_3]
    else:
        if3d, filter_size, filter_num, l_rate, load_episode, inception, node_1, node_2 = fulllist[2:]
    args = [filter_size, filter_num, l_rate, load_episode]
    tf.reset_default_graph()
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())
        model = Seq_deepCpf1(filter_size, filter_num, node_1, node_2, args[2])
        
        saver = tf.train.Saver()
        saver.restore(sess, best_model_path + best_model)
        TEST_Y = []
        
        for i in range(len(TEST_NUM_SET)):
            print ("TEST_NUM : {}".format(TEST_NUM_SET[i]))
            
            Model_Finaltest(sess, TEST_X[i], filter_size, filter_num, if3d, model, args, load_episode, best_model_path)
