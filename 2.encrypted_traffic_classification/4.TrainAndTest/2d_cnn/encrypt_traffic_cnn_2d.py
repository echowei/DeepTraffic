# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

import time
import sys
import numpy as npß
import os

# load MNIST data
import input_data

# start tensorflow interactiveSession
import tensorflow as tf

# Note: if class numer is 2 or 20, please edit the variable named "num_classes" in /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py"
DATA_DIR = sys.argv[1]
CLASS_NUM = int(sys.argv[2])
TRAIN_ROUND = int(sys.argv[3])
#DATA_DIR = '/root/data/withip/10class/BenignFlowAllLayers'
#CLASS_NUM = 10
#TRAIN_ROUND = 40000

dict_2class = {0:'Novpn',1:'Vpn'}
dict_6class_novpn = {0:'Chat',1:'Email',2:'File',3:'P2p',4:'Streaming',5:'Voip'}
dict_6class_vpn = {0:'Vpn_Chat',1:'Vpn_Email',2:'Vpn_File',3:'Vpn_P2p',4:'Vpn_Streaming',5:'Vpn_Voip'}
dict_12class = {0:'Chat',1:'Email',2:'File',3:'P2p',4:'Streaming',5:'Voip',6:'Vpn_Chat',7:'Vpn_Email',8:'Vpn_File',9:'Vpn_P2p',10:'Vpn_Streaming',11:'Vpn_Voip'}
dict = {}

folder = os.path.split(DATA_DIR)[1]

sess = tf.InteractiveSession()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# function: find a element in a list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create the model
# placeholder
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, CLASS_NUM])

# first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, CLASS_NUM])
b_fc2 = bias_variable([CLASS_NUM])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# define var&op of training&testing
actual_label = tf.argmax(y_, 1)
label,idx,count = tf.unique_with_counts(actual_label)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
predict_label = tf.argmax(y_conv, 1)
label_p,idx_p,count_p = tf.unique_with_counts(predict_label)
correct_prediction = tf.equal(predict_label, actual_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
correct_label=tf.boolean_mask(actual_label,correct_prediction)
label_c,idx_c,count_c=tf.unique_with_counts(correct_label)

# if model exists: restore it
# else: train a new model and save it
saver = tf.train.Saver()
model_name = "model_" + str(CLASS_NUM) + "class_" + folder
model =  model_name + '/' + model_name + ".ckpt"
if not os.path.exists(model + ".meta"):
    sess.run(tf.global_variables_initializer())
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    for i in range(TRAIN_ROUND+1):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
            s = "step %d, train accuracy %g" %(i, train_accuracy)
            print s
            # if i%2000 == 0:
            #     with open('out.txt','a') as f:
            #         f.write(s + "\n")
        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    
    save_path = saver.save(sess, model)
    print("Model saved in file:", save_path)
else:     
    saver.restore(sess, model)
    print("Model restored: " + model)
    
# evaluate the model
if CLASS_NUM == 12:
    dict = dict_12class
elif CLASS_NUM == 2:
    dict = dict_2class
elif CLASS_NUM == 6:
    if folder.startswith('Novpn'):
        dict = dict_6class_novpn
    elif folder.startswith('Vpn'):
        dict = dict_6class_vpn
label,count,label_p,count_p,label_c,count_c,acc=sess.run([label,count,label_p,count_p,label_c,count_c,accuracy],{x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})
acc_list = []
for i in range(CLASS_NUM):
    n1 = find_element_in_list(i,label.tolist())
    count_actual = count[n1]
    n2 = find_element_in_list(i,label_c.tolist())
    count_correct = count_c[n2] if n2>-1 else 0
    n3 = find_element_in_list(i,label_p.tolist())
    count_predict = count_p[n3] if n3>-1 else 0
    
    recall = float(count_correct)/float(count_actual)
    precision = float(count_correct)/float(count_predict) if count_predict>0 else -1
    acc_list.append([str(i),dict[i],str(recall),str(precision)])
with open('out.txt','a') as f:
    f.write("\n")
    t = time.strftime('%Y-%m-%d %X',time.localtime())
    f.write(t + "\n")
    f.write('DATA_DIR: ' + DATA_DIR+ "\n")
    for item in acc_list:
        f.write(', '.join(item) + "\n")
    f.write('Total accuracy: ' + str(acc) + "\n\n")
