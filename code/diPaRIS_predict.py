import math
import keras_nlp
import numpy as np
from sklearn.model_selection import KFold
import sklearn
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.layers import Activation, Concatenate, Bidirectional, GRU, Input, Dropout
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
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

def icshapeDS(seq, struc):
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
        if struc[i] != -1 and struc[i + 1] != -1:
            if struc[i] >= struc[i + 1]:
                vector[i][0] = 1
            if struc[i] < struc[i + 1]:
                vector[i][1] = 1
            if struc[i] >= 0.233:
                vector[i][4] = 1
            if struc[i + 1] >= 0.233:
                vector[i][5] = 1

        if struc[i] != -1 and struc[i + 1] == -1:
            vector[i][2] = 1
            if struc[i] >= 0.233:
                vector[i][4] = 1
        if struc[i] == -1 and struc[i + 1] != -1:
            vector[i][3] = 1
            if struc[i + 1] >= 0.233:
                vector[i][5] = 1

        if struc[i] == -1 and struc[i + 1] == -1:
            for j in range(0, 6):
                vector[i][j] = -1

    return vector

def dealwithdata(seq, struc):
    dataX = []
    dataX2 = []
    dataX.append(coden(seq))
    dataX2.append(icshapeDS(seq, struc))
    
    dataX = np.array(dataX)
    dataX = dataX[:, np.newaxis, :]
    dataX2 = np.array(dataX2)
    dataX2 = dataX2[:, np.newaxis, :]

    return dataX, dataX2

def diPaRIS():
    # input
    left_input = Input(shape=(1, 101, 4), name='left_input')
    right_input = Input(shape=(1, 100, 7), name='right_input')
    left_conv = keras.layers.Conv2D(64, (6, 4), padding="same")(left_input)
    left_norm = keras.layers.BatchNormalization()(left_conv)
    left_act = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                  shared_axes=None)(left_norm)
    left_sq = keras.backend.squeeze(left_act, axis=1)
    left_bilstm = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(left_sq)
    left_SE = keras.layers.multiply([left_act, left_bilstm])

    right_conv = keras.layers.Conv2D(64, (10, 7), padding="same")(right_input)
    right_norm = keras.layers.BatchNormalization()(right_conv)
    right_act = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                   shared_axes=None)(right_norm)
    right_sq = keras.backend.squeeze(right_act, axis=1)
    right_bilstm = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(right_sq)
    right_SE = keras.layers.multiply([right_act, right_bilstm])

    merge = keras.layers.Concatenate(axis=-2)([left_SE, right_SE])
    # down-sample
    A1 = keras.layers.Conv2D(32, (3, 3), padding="same")(merge)
    A1 = keras.layers.Activation('relu')(A1)
    A1 = keras.layers.Conv2D(32, (3, 3), padding="same")(A1)
    A1 = keras.layers.Activation('relu')(A1)
    A2 = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(A1)
    A2 = keras.layers.BatchNormalization()(A2)
    A2 = keras.layers.Dropout(0.3)(A2)

    A3 = keras.layers.Conv2D(64, (3, 3), padding="same")(A2)
    A3 = keras.layers.Activation('relu')(A3)
    A3 = keras.layers.Conv2D(64, (3, 3), padding="same")(A3)
    A3 = keras.layers.Activation('relu')(A3)
    A4 = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(A3)
    A4 = keras.layers.BatchNormalization()(A4)
    A4 = keras.layers.Dropout(0.3)(A4)

    A5 = keras.layers.Conv2D(128, (3, 3), padding="same")(A4)
    A5 = keras.layers.Activation('relu')(A5)
    A5 = keras.layers.Conv2D(128, (3, 3), padding="same")(A5)
    A5 = keras.layers.Activation('relu')(A5)
    A6 = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(A5)
    A6 = keras.layers.BatchNormalization()(A6)
    A6 = keras.layers.Dropout(0.3)(A6)
    # transformer
    A1 = keras.backend.squeeze(A1, axis=1)
    A11 = keras_nlp.layers.TransformerEncoder(32, 32, 0.3)(A1)
    A11 = keras_nlp.layers.TransformerEncoder(32, 32, 0.3)(A11)
    A11 = keras_nlp.layers.TransformerEncoder(32, 32, 0.3)(A11)
    A11 = keras.layers.multiply([A1, A11])
    A3 = keras.backend.squeeze(A3, axis=1)
    A13 = keras_nlp.layers.TransformerEncoder(64, 64, 0.3)(A3)
    A13 = keras_nlp.layers.TransformerEncoder(64, 64, 0.3)(A13)
    A13 = keras_nlp.layers.TransformerEncoder(64, 64, 0.3)(A13)
    A13 = keras.layers.multiply([A3, A13])
    A5 = keras.backend.squeeze(A5, axis=1)
    A15 = keras_nlp.layers.TransformerEncoder(128, 128, 0.3)(A5)
    A15 = keras_nlp.layers.TransformerEncoder(128, 128, 0.3)(A15)
    A15 = keras_nlp.layers.TransformerEncoder(128, 128, 0.3)(A15)
    A15 = keras.layers.multiply([A5, A15])

    A7 = keras.layers.Conv2D(256, (3, 3), padding="same")(A6)
    A7 = keras.layers.Activation('relu')(A7)
    A7 = keras.layers.Conv2D(256, (3, 3), padding="same")(A7)
    A7 = keras.layers.Activation('relu')(A7)
    A8 = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(A7)
    A8 = keras.layers.BatchNormalization()(A8)
    A8 = keras.layers.Dropout(0.3)(A8)

    A1 = keras.backend.squeeze(A2, axis=1)
    A11 = keras_nlp.layers.TransformerEncoder(32, 32, 0.3)(A1)
    A11 = keras_nlp.layers.TransformerEncoder(32, 32, 0.3)(A11)
    A11 = keras_nlp.layers.TransformerEncoder(32, 32, 0.3)(A11)
    A11 = keras.layers.multiply([A1, A11])
    A3 = keras.backend.squeeze(A3, axis=1)
    A13 = keras_nlp.layers.TransformerEncoder(64, 64, 0.3)(A3)
    A13 = keras_nlp.layers.TransformerEncoder(64, 64, 0.3)(A13)
    A13 = keras_nlp.layers.TransformerEncoder(64, 64, 0.3)(A13)
    A13 = keras.layers.multiply([A3, A13])
    A5 = keras.backend.squeeze(A5, axis=1)
    A15 = keras_nlp.layers.TransformerEncoder(128, 128, 0.3)(A5)
    A15 = keras_nlp.layers.TransformerEncoder(128, 128, 0.3)(A15)
    A15 = keras_nlp.layers.TransformerEncoder(128, 128, 0.3)(A15)
    A15 = keras.layers.multiply([A5, A15])

    inputs = keras.layers.Concatenate(axis=-2)([A1, A3, A5])
    # final
    outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)
    model = keras.Model(inputs=[left_input, right_input], outputs=outputs)

    return model

def main(seq, struc):
    dataX, dataX2 = dealwithdata(seq, struc)
    
    # Define the model
    model = diPaRIS()
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit([dataX, dataX2], dataY, epochs=100, batch_size=32, validation_split=0.2)

if __name__ == "__main__":
    # Example sequences and structures
    example_seq = "TGTTGATTTTATTTGACCCCTGGAGTGGTGGGTCTCATCTTTCCCATCTCGCCTGAGAGCGGCTGAGGGCTGCCTCACTGCAAATCCTCCCCACAGCGTCA"
    example_struc = [0.213,0.298,0.27,0.313,0.355,0.253,0.421,0.562,0.8,0.934,0.877,1,0.952,0.529,0.208,0.026,0.003,0.04,0.095,0.123,0.26,0.59,0.984,0.906,1,0.272,0.258,0.56,0.002,0.313,0.202,0.366,0.447,0.611,0.97,1,1,0.702,0.972,1,1,0.35,0.215,0.391,0.567,0.595,0.31,0.757,0.73,0.54,0.541,0.134,0.31,0.433,0.46,0.161,1,0.393,0.515,0.665,0.832,0.45,0.921,0.572,0.245,0.205,0.617,0.164,0.136,0.246,0.193,0.055,0.082,0.068,0.068,0.152,0.194,0.291,0.18,0.208,0.348,0.71,0.64,0.473,0.306,0.21,0.168,0.182,0.029,0.084,0.183,0.169,0.281,0.379,0.281,0.339,0.24,0.465,0.128,0.198,0.242
]  # Example structure data
    main(example_seq, example_struc)
