# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D, Activation
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K
import numpy as np
import tensorflow as tf
import keras.callbacks
import sys
import os
import cPickle as pickle
from timeit import default_timer as timer
import glob
import time

LSTM_UNITS = 92
MINI_BATCH = 10
TRAIN_STEPS_PER_EPOCH = 12000
VALIDATION_STEPS_PER_EPOCH = 800
DATA_DIR = '/root/data/PreprocessedISCX2012_5class_pkl/'
CHECKPOINTS_DIR = './iscx2012_cnn_rnn_5class_new_checkpoints/'

dict_5class = {0:'Normal', 1:'BFSSH', 2:'Infilt', 3:'HttpDoS', 4:'DDoS'}

def update_confusion_matrix(confusion_matrix, actual_lb, predict_lb):
    for idx, value in enumerate(actual_lb):
        p_value = predict_lb[idx]
        confusion_matrix[value, p_value] += 1
    return confusion_matrix

# function: find an element in a list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1

def truncate(f, n):
    trunc_f = np.math.floor(f * 10 ** n) / 10 ** n
    return '{:.2f}'.format(trunc_f) # only for 0.0 => 0.00

def binarize(x, sz=256):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 256

def byte_block(in_layer, nb_filter=(64, 100), filter_length=(3, 3), subsample=(2, 1), pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):
        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)
        # block = BatchNormalization()(block)
        # block = Dropout(0.1)(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    # block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    block = GlobalMaxPool1D()(block)
    block = Dense(128, activation='relu')(block)
    return block

def mini_batch_generator(sessions, labels, indices, batch_size):
    Xbatch = np.ones((batch_size, PACKET_NUM_PER_SESSION, PACKET_LEN), dtype=np.int64) * -1
    Ybatch = np.ones((batch_size,5), dtype=np.int64) * -1
    batch_idx = 0
    while True:
        for idx in indices:
            for i, packet in enumerate(sessions[idx]):
                if i < PACKET_NUM_PER_SESSION:
                    for j, byte in enumerate(packet[:PACKET_LEN]):
                        Xbatch[batch_idx, i, (PACKET_LEN - 1 - j)] = byte            
            Ybatch[batch_idx] = np_utils.to_categorical(labels[idx], num_classes=5)[0]
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (Xbatch, Ybatch)

# read argv
print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])        
# if len(sys.argv) == 5:
#     if os.path.exists(str(sys.argv[4])):
#         print ("Checkpoint : %s" % str(sys.argv[4]))
#         checkpoint = str(sys.argv[4])

# read sessions and labels from pickle files
sessions = []
labels = []
t1 = timer()
num_pkls = len(glob.glob(DATA_DIR + 'ISCX2012_labels_*.pkl'))
for i in range(num_pkls):
    session_pkl = DATA_DIR + 'ISCX2012_pcaps_' + str(i) + '.pkl'
    session_lists  = pickle.load(open(session_pkl, 'rb'))
    sessions.extend(session_lists)
    label_pkl = DATA_DIR + 'ISCX2012_labels_' + str(i) + '.pkl'
    label_lists = pickle.load(open(label_pkl, 'rb'))    
    labels.extend(label_lists)    
    print i
t2 = timer()
print t2 - t1
print('Sample doc{}'.format(sessions[1200]))
labels = np.array(labels)

# arg_list = [[50,6],[100,6],[200,6],[300,6],[400,6],[500,6],[600,6],[700,6],[800,6],[900,6],[1000,6],
#             [100,8],[100,10],[100,12],[100,14],[100,16],[100,18],[100,20],[100,22],[100,24],[100,26],[100,28],[100,30]]
# arg_list = [[100,12],[100,14],[100,16],[100,18],[100,20],[100,22],[100,24],[100,26],[100,28],[100,30]]
arg_list = [[600,14],[700,14],[800,14],[900,14],[1000,14]]
# arg_list = [[100,6]]
for arg in arg_list:
    PACKET_LEN = arg[0]
    PACKET_NUM_PER_SESSION = arg[1]
    TRAIN_EPOCHS = 8

    # create train/validate data generator 
    normal_indices = np.where(labels == 0)[0]
    attack_indices = [np.where(labels == i)[0] for i in range(1,5)]
    print len(normal_indices)
    print len(attack_indices)
    test_normal_indices = np.random.choice(normal_indices, int(len(normal_indices)*0.4))
    test_attack_indices = np.concatenate([np.random.choice(attack_indices[i], int(len(attack_indices[i])*0.4)) for i in range(4)])
    test_indices = np.concatenate([test_normal_indices, test_attack_indices]).astype(int)
    train_indices = np.array(list(set(np.arange(len(labels))) - set(test_indices)))
    train_data_generator  = mini_batch_generator(sessions, labels, train_indices, MINI_BATCH)
    val_data_generator    = mini_batch_generator(sessions, labels, test_indices, MINI_BATCH)
    test_data_generator   = mini_batch_generator(sessions, labels, test_indices, MINI_BATCH)

    # create model
    session = Input(shape=(PACKET_NUM_PER_SESSION, PACKET_LEN), dtype='int64')
    input_packet = Input(shape=(PACKET_LEN,), dtype='int64')
    embedded = Lambda(binarize, output_shape=binarize_outshape)(input_packet)
    block2 = byte_block(embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
    block3 = byte_block(embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))
    packet_encode = concatenate([block2, block3], axis=-1)
    # packet_encode = Dropout(0.2)(packet_encode)
    encoder = Model(inputs=input_packet, outputs=packet_encode)
    encoder.summary()
    encoded = TimeDistributed(encoder)(session)
    lstm_layer = LSTM(LSTM_UNITS, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
    lstm_layer2 = LSTM(LSTM_UNITS, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0)(lstm_layer)
    dense_layer = Dense(5, name='dense_layer')(lstm_layer2)
    output = Activation('softmax')(dense_layer)
    model = Model(outputs=output, inputs=session)
    model.summary()

    # if input checkpoint, test with saved model and save predicted results
    if checkpoint:
        model.load_weights(checkpoint)
        sub_model = Model(inputs=model.input,
                        outputs=model.get_layer('dense_layer').output)
        test_steps = np.math.ceil(float(len(test_indices)) / MINI_BATCH)
        embd = sub_model.predict_generator(test_data_generator, steps=test_steps)
        print type(embd)
        print embd.shape
        print len(embd)
        print embd[0]
        np.save('./embeddings_iscx2012.npy', embd)
        np.save('./labels_iscx2012.npy', labels[test_indices])
        break

    # train and validate model
    script_name = os.path.basename(sys.argv[0]).split('.')[0]
    weight_file = CHECKPOINTS_DIR + script_name + '_' + str(PACKET_LEN) + '_' + str(PACKET_NUM_PER_SESSION) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
    check_cb = keras.callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min')
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    train_time = 0
    start_train = timer()
    model.fit_generator(
        generator=train_data_generator, 
        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
        epochs=TRAIN_EPOCHS,
        callbacks=[check_cb, earlystop_cb],
        validation_data=val_data_generator,
        validation_steps=VALIDATION_STEPS_PER_EPOCH)
    end_train = timer()
    train_time = end_train - start_train

    # test model
    start_test = timer()
    test_steps = np.math.ceil(float(len(test_indices)) / MINI_BATCH)
    predictions = model.predict_generator(test_data_generator, steps=test_steps)
    end_test = timer()
    test_time = end_test - start_test

    # stat and save
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = labels[test_indices]
    if len(predicted_labels) > len(true_labels):
        num_pad = len(predicted_labels) - len(true_labels)
        true_labels = np.concatenate([true_labels, true_labels[0:num_pad]])
    print len(predicted_labels)
    print len(true_labels)
    len_test = len(true_labels)
    cf_ma = np.zeros((5,5), dtype=int)
    update_confusion_matrix(cf_ma, true_labels, predicted_labels)
    metrics_list = []
    for i in range(5):
        if i == 0:
            metrics_list.append([dict_5class[i], str(i), str(cf_ma[i,0]), str(cf_ma[i,1]), str(cf_ma[i,2]), str(cf_ma[i,3]), str(cf_ma[i,4]), '--', '--', '--'])
        else:
            acc = truncate((float(len_test-cf_ma[:,i].sum()-cf_ma[i,:].sum()+cf_ma[i,i]*2)/len_test)*100, 2)
            tpr = truncate((float(cf_ma[i,i])/cf_ma[i].sum())*100, 2)
            fpr = truncate((float(cf_ma[0,i])/cf_ma[0].sum())*100, 2)
            metrics_list.append([dict_5class[i], str(i), str(cf_ma[i,0]), str(cf_ma[i,1]), str(cf_ma[i,2]), str(cf_ma[i,3]), str(cf_ma[i,4]), str(acc), str(tpr), str(fpr)])
    overall_acc = truncate((float(cf_ma[0,0]+cf_ma[1,1]+cf_ma[2,2]+cf_ma[3,3]+cf_ma[4,4])/len_test)*100, 2)
    overall_tpr = truncate((float(cf_ma[1,1]+cf_ma[2,2]+cf_ma[3,3]+cf_ma[4,4])/cf_ma[1:].sum())*100, 2)
    overall_fpr = truncate((float(cf_ma[0,1:].sum())/cf_ma[0,:].sum())*100, 2)
    with open('iscx12_cnn_rnn_5class_new.txt','a') as f:
        f.write("\n")
        t = time.strftime('%Y-%m-%d %X',time.localtime())
        f.write(t + "\n")
        f.write('CLASS_NUM: 5\n')
        f.write('PACKET_LEN: ' + str(PACKET_LEN) + "\n")
        f.write('PACKET_NUM_PER_SESSION: ' + str(PACKET_NUM_PER_SESSION) + "\n")
        f.write('MINI_BATCH: ' + str(MINI_BATCH) + "\n")
        f.write('TRAIN_EPOCHS: ' + str(TRAIN_EPOCHS) + "\n")
        f.write('DATA_DIR: ' + DATA_DIR + "\n")
        f.write("label\tindex\t0\t1\t2\t3\t4\tACC\tTPR\tFPR\n")
        for metrics in metrics_list:
            f.write('\t'.join(metrics) + "\n")
        f.write('Overall accuracy: ' + str(overall_acc) + "\n")
        f.write('Overall TPR: ' + str(overall_tpr) + "\n")
        f.write('Overall FPR: ' + str(overall_fpr) + "\n")
        f.write('Train time(second): ' + str(int(train_time)) + "\n")
        f.write('Test time(second): ' + str(int(test_time)) + "\n\n")