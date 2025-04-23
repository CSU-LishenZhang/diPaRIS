import math
import keras_nlp
import numpy as np
from sklearn.model_selection import KFold
import sklearn
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.layers import Activation, Concatenate, Bidirectional, GRU
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
import argparse
from ePooling import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
import matplotlib as mpl

mpl.use('Agg')

def coden(seq):
    dict_1mer = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    vectors = np.zeros((len(seq), 4))
    for i in range(len(seq)):
        vectors[i][dict_1mer[seq[i].replace('T', 'U')]] = 1
    return vectors

def chunks_two(seq, win):
    seqlen = len(seq)
    for i in range(seqlen):
        j = seqlen if i + win > seqlen else i + win
        yield seq[i:j]
        if j == seqlen: break
    return

def icshapeDS(seq, icshape):
    probabilities = {}
    kmer = {}
    total_windows = len(seq) - 1
    for subseq in chunks_two(seq, 2):
        if subseq in kmer:
            kmer[subseq] = kmer[subseq] + 1
        else:
            kmer[subseq] = 1
    for key in kmer:
        p = (kmer[key] / total_windows)
        probabilities[key] = -(p * math.log(p, 2))
    vector = np.zeros((len(seq) - 1, 7))
    for i in range(0, len(seq) - 1):
        vector[i][6] = probabilities[seq[i:i + 2]]
        if icshape[i] != -1 and icshape[i + 1] != -1:
            if icshape[i] >= icshape[i + 1]:
                vector[i][0] = 1
            if icshape[i] < icshape[i + 1]:
                vector[i][1] = 1
            if icshape[i] >= 0.233:
                vector[i][4] = 1
            if icshape[i + 1] >= 0.233:
                vector[i][5] = 1

        if icshape[i] != -1 and icshape[i + 1] == -1:
            vector[i][2] = 1
            if icshape[i] >= 0.233:
                vector[i][4] = 1
        if icshape[i] == -1 and icshape[i + 1] != -1:
            vector[i][3] = 1
            if icshape[i + 1] >= 0.233:
                vector[i][5] = 1

        if icshape[i] == -1 and icshape[i + 1] == -1:
            for j in range(0, 6):
                vector[i][j] = -1

    return vector

def dealwithdata(protein):
    seqP = []
    seqN = []
    dataX = []
    dataX2 = []
    icshapeP = []
    icshapeN = []
    dataY = []
    with open('../dataset/' + protein + '/positive_seq') as f:
        for line in f:
            if '>' not in line:
                seqP.append(line.strip())
                dataY.append([0, 1])
    with open('../dataset/' + protein + '/positive_str') as f:
        for line in f:
            row = []
            lines = line.strip().split("\t")
            for x in lines:
                row.append(float(x))
            icshapeP.append(row)
    for i in range(len(icshapeP)):
        dataX.append(coden(seqP[i]))
        dataX2.append(icshapeDS(seqP[i], icshapeP[i]))

    with open('../dataset/' + protein + '/negative_seq') as f:
        for line in f:
            if '>' not in line:
                seqN.append(line.strip())
                dataY.append([1, 0])
    with open('../dataset/' + protein + '/negative_str') as f:
        for line in f:
            row = []
            lines = line.strip().split("\t")
            for x in lines:
                row.append(float(x))
            icshapeN.append(row)
    for i in range(len(icshapeN)):
        dataX.append(coden(seqN[i]))
        dataX2.append(icshapeDS(seqN[i], icshapeN[i]))

    indexes = np.random.choice(len(dataY), len(dataY), replace=False)
    dataX = np.array(dataX)[indexes]
    dataX = dataX[:, np.newaxis, :]
    dataY = np.array(dataY)[indexes]
    dataX2 = np.array(dataX2)[indexes]
    dataX2 = dataX2[:, np.newaxis, :]
    train_X = np.array(dataX)[round(len(indexes) / 5):]
    test_X = np.array(dataX)[:round(len(indexes) / 5)]
    train_y = np.array(dataY)[round(len(indexes) / 5):]
    test_y = np.array(dataY)[:round(len(indexes) / 5)]
    train_X2 = np.array(dataX2)[round(len(indexes) / 5):]
    test_X2 = np.array(dataX2)[:round(len(indexes) / 5)]

    return train_X, test_X, train_y, test_y, train_X2, test_X2
    
INITIALIZER = keras.initializers.HeNormal()  
REGULARIZER = keras.regularizers.L2(1e-3)

def conv2d_block(x, filters):
    # first layer
    shortcut = keras.layers.Conv2D(filters, 1, padding="same", kernel_initializer=INITIALIZER, 
        )(x)
    reduced_filters = filters // 4
    x = keras.layers.Conv2D(reduced_filters, 1, padding="same", 
        kernel_regularizer=REGULARIZER)(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Conv2D(reduced_filters, 3, padding="same", kernel_initializer=INITIALIZER, 
        )(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Conv2D(reduced_filters, 3, padding="same", 
        kernel_regularizer=REGULARIZER)(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Conv2D(filters, 1, padding="same", kernel_initializer=INITIALIZER, 
        )(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.add([shortcut, x])
    return x

def conv1d_block(x, filters):
    shortcut = keras.layers.Conv1D(filters, 1, padding="same", kernel_initializer=INITIALIZER, 
        )(x)
    reduced_filters = filters // 4
    x = keras.layers.Conv1D(reduced_filters, 1, padding="same", 
        kernel_regularizer=REGULARIZER)(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(reduced_filters, 3, padding="same", kernel_initializer=INITIALIZER, 
        )(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Conv1D(reduced_filters, 3, padding="same", 
        kernel_regularizer=REGULARIZER)(x)
    x = keras.layers.PReLU()(x)
    y = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(filters, 1, padding="same", kernel_initializer=INITIALIZER, 
        kernel_regularizer=REGULARIZER)(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.add([shortcut, x])
    return x

def diPaRIS():
    #input
    left_input = keras.layers.Input(shape=(1, 101, 4), name='left_input')
    right_input = keras.layers.Input(shape=(1, 100, 7), name='right_input')
    left_conv = keras.layers.Conv2D(64, (10, 4), padding="same", kernel_initializer=INITIALIZER, 
        kernel_regularizer=REGULARIZER)(left_input)
    left_norm = keras.layers.BatchNormalization()(left_conv)
    left_act = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(left_norm)
    left_sq = keras.backend.squeeze(left_act, axis=1)
    left_bilstm = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(left_sq)
    left_bilstm = keras.layers.Reshape((1, 101, 64))(left_bilstm)
    left_SE = keras.layers.multiply([left_act, left_bilstm])
    left_SE = keras.layers.SpatialDropout2D(0.3)(left_SE)

    right_conv = keras.layers.Conv2D(64, (16, 7), padding="same", kernel_initializer=INITIALIZER, 
        kernel_regularizer=REGULARIZER)(right_input)
    right_norm = keras.layers.BatchNormalization()(right_conv)
    right_act = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(right_norm)
    right_sq = keras.backend.squeeze(right_act, axis=1)
    right_bilstm = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(right_sq)
    right_bilstm = keras.layers.Reshape((1, 100, 64))(right_bilstm)
    right_SE = keras.layers.multiply([right_act, right_bilstm])
    right_SE = keras.layers.SpatialDropout2D(0.3)(right_SE)

    merge = keras.layers.Concatenate(axis=2)([left_SE, right_SE])

    # #down-sample
    A1 = conv2d_block(merge, 32)
    A2 = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(A1)
    A2 = keras.layers.BatchNormalization()(A2)
    A2 = keras.layers.SpatialDropout2D(0.3)(A2)

    A3 = conv2d_block(A2, 64)
    A4 = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(A3)
    A4 = keras.layers.BatchNormalization()(A4)
    A4 = keras.layers.SpatialDropout2D(0.3)(A4)

    A5 = conv2d_block(A4, 128)
    A6 = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(A5)
    A6 = keras.layers.BatchNormalization()(A6)
    A6 = keras.layers.SpatialDropout2D(0.3)(A6)
    # #transformer
    A1 = keras.backend.squeeze(A1, axis=1)
    A11 = keras.layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.3)(A1, A1)
    A11 = keras.layers.multiply([A1, A11])
    A3 = keras.backend.squeeze(A3, axis=1)
    A13 = keras.layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.3)(A3, A3)
    A13 = keras.layers.multiply([A3, A13])
    A5 = keras.backend.squeeze(A5, axis=1)
    A15 = keras.layers.MultiHeadAttention(num_heads=1, key_dim=128, dropout=0.3)(A5, A5)
    A15 = keras.layers.multiply([A5, A15])
    #bottle-neck
    A7 = conv2d_block(A6, 256)
    
    A7 = keras.backend.squeeze(A7, axis=1)
    A7 = conv1d_block(A7, 256)
    # #up-sample
    A8 = keras.layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding="same")(A7)
    A8 = keras.layers.LayerNormalization()(A8)
    A8 = keras.layers.Activation('relu')(A8)
    A8 = keras.layers.Concatenate(axis=-1)([A8, A15])
    A8 = conv1d_block(A8, 128)
    A8 = keras.layers.LayerNormalization()(A8)

    A9 = keras.layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding="same")(A8)
    A9 = keras.layers.LayerNormalization()(A9)
    A9 = keras.layers.Activation('relu')(A9)
    A9 = keras.layers.Concatenate(axis=-1)([A9, A13])
    A9 = conv1d_block(A9, 64)
    A9 = keras.layers.LayerNormalization()(A9)

    A0 = keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding="valid")(A9)
    A0 = keras.layers.LayerNormalization()(A0)
    A0 = keras.layers.Activation('relu')(A0)
    A0 = keras.layers.Concatenate(axis=-1)([A0, A11])
    A = conv1d_block(A0, 32)
    #classify
    stack1 = keras.layers.LayerNormalization()(A)
    stack2 = keras.layers.AveragePooling1D(pool_size=int(stack1.shape[1]))(stack1)
    stack3 = keras.layers.AveragePooling1D(40)(stack1)
    stack4 = keras.layers.AveragePooling1D(8)(stack1)
    stack6 = keras.layers.Concatenate(axis=1)([stack2, stack3, stack4])
    shortcut = GlobalExpectationPooling1D(mode=0, m_trainable=False, m_value=1)(stack6)
    shortcut = keras.layers.Dense(16,
                                kernel_regularizer=REGULARIZER
                                )(shortcut)
    shortcut = keras.layers.Dense(8,
                                kernel_initializer=keras.initializers.GlorotUniform(),
                                # kernel_regularizer=REGULARIZER
                                )(shortcut)
    shortcut = keras.layers.Dense(4,
                                kernel_regularizer=REGULARIZER
                                )(shortcut)
    output = keras.layers.Dense(2, activation="softmax", 
                                kernel_initializer=keras.initializers.GlorotUniform(),
                                # kernel_regularizer=REGULARIZER
                                )(shortcut)
    return Model(inputs=[left_input, right_input], outputs=[output])

def main(protein_list):
    for protein in protein_list:
        print(protein)
        trainXeval, test_X, trainYeval, test_y, train_X2, test_X2 = dealwithdata(protein)
        test_y = test_y[:, 1]

        kf = KFold(n_splits=5).split(trainYeval)
        auc_list = []
        acc_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        aupr_list = []

        for train_index, eval_index in kf:
            train_X = trainXeval[train_index]
            train_y = trainYeval[train_index]
            eval_X = trainXeval[eval_index]
            eval_y = trainYeval[eval_index]

            model = diPaRIS()
            print(model.summary())
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Load previous weights if available
            if os.path.exists(f'../model/diPaRIS_{protein}.h5'):
                print(f"Loading previous best weights for model: {protein}")
                model.load_weights(f'../model/diPaRIS_{protein}.h5')

            def step_decay(epoch):
                initial_lrate = 0.0005
                drop = 0.8
                epochs_drop = 5.0
                lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                print(lrate)
                return lrate

            callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
                         LearningRateScheduler(step_decay)]
            history = model.fit(train_X, train_y, batch_size=16, epochs=64, verbose=0, validation_data=(eval_X, eval_y),
                                callbacks=callbacks)
            model.save(f'../model/diPaRIS_{protein}.h5')

            prediction = model.predict(test_X)[:, 1]
            aucs = roc_auc_score(test_y, prediction)
            auc_list.append(aucs)
            predictions = [round(i, 0) for i in prediction]
            acc = sklearn.metrics.accuracy_score(test_y, predictions)
            acc_list.append(acc)
            precision = sklearn.metrics.precision_score(test_y, predictions)
            precision_list.append(precision)
            recall = sklearn.metrics.recall_score(test_y, predictions)
            recall_list.append(recall)
            f1_score = sklearn.metrics.f1_score(test_y, predictions)
            f1_score_list.append(f1_score)
            pre_vals, recall_vals, thresholds2 = precision_recall_curve(test_y, predictions)
            aupr_list.append(auc(recall_vals, pre_vals))

        print(f"AUC: {auc_list}", protein)
        print(f"ACC: {acc_list}", protein)
        print(f"Precision: {precision_list}", protein)
        print(f"Recall: {recall_list}", protein)
        print(f"F1-score: {f1_score_list}", protein)
        print(f"AUPR: {aupr_list}", protein)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diPaRIS model on specified datasets.")
    
    # Default dataset AKAP1-HepG2, can be overridden by user input
    parser.add_argument(
        '-d', '--datasets', 
        nargs='+', 
        default=['AKAP1-HepG2'],  # Set default dataset
        help='List of dataset names to train on (e.g., -d AKAP1-HepG2 YourDataset1 YourDataset2)'
    )

    args = parser.parse_args()
    main(protein_list=args.datasets)

