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
    dataX = []
    dataX2 = []
    icshapeP = []
    with open('../dataset/' + protein + '/positive_seq') as f:
        for line in f:
            if '>' not in line:
                seqP.append(line.strip())
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

    dataX = np.array(dataX)[indexes]
    dataX = dataX[:, np.newaxis, :]
    dataX2 = np.array(dataX2)[indexes]
    dataX2 = dataX2[:, np.newaxis, :]

    return dataX, dataX2

def diPaRIS():
    # input
    left_input = keras.layers.Input(shape=(1, 101, 768), name='left_input')
    right_input = keras.layers.Input(shape=(1, 101, 768), name='right_input')
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

    A7 = keras.backend.squeeze(A7, axis=1)
    A7 = keras.layers.Conv1D(filters=256, kernel_size=3, padding="same")(A7)
    A7 = keras.layers.Activation('relu')(A7)
    A7 = keras.layers.Conv1D(filters=256, kernel_size=3, padding="same")(A7)
    A7 = keras.layers.Activation('relu')(A7)

    # up-sample
    A8 = keras.layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding="same")(A7)
    A8 = keras.layers.LayerNormalization()(A8)
    A8 = keras.layers.Activation('relu')(A8)
    A8 = Concatenate(axis=-1)([A8, A15])
    A8 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(A8)
    A8 = keras.layers.Activation('relu')(A8)
    A8 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(A8)
    A8 = keras.layers.Activation('relu')(A8)
    A8 = keras.layers.LayerNormalization()(A8)

    A9 = keras.layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding="valid")(A8)
    A9 = keras.layers.LayerNormalization()(A9)
    A9 = keras.layers.Activation('relu')(A9)
    A9 = Concatenate(axis=-1)([A9, A13])
    A9 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(A9)
    A9 = keras.layers.Activation('relu')(A9)
    A9 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(A9)
    A9 = keras.layers.Activation('relu')(A9)
    A9 = keras.layers.LayerNormalization()(A9)

    A0 = keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding="same")(A9)
    A0 = keras.layers.LayerNormalization()(A0)
    A0 = keras.layers.Activation('relu')(A0)
    A0 = Concatenate(axis=-1)([A0, A11])
    A0 = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same")(A0)
    A0 = keras.layers.Activation('relu')(A0)
    A0 = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same")(A0)
    A = keras.layers.Activation('relu')(A0)
    # classify
    stack1 = keras.layers.LayerNormalization()(A)
    stack2 = keras.layers.AveragePooling1D(pool_size=int(stack1.shape[1]))(stack1)
    stack3 = keras.layers.AveragePooling1D(40)(stack1)
    stack4 = keras.layers.AveragePooling1D(8)(stack1)
    stack6 = Concatenate(axis=1)([stack2, stack3, stack4])
    stack7 = GlobalExpectationPooling1D(mode=0, m_trainable=False, m_value=1)(stack6)
    output = keras.layers.Dense(2, activation="softmax")(stack7)
    return Model(inputs=[left_input, right_input], outputs=[output])

def main():
    parser = argparse.ArgumentParser(description="Load a trained U_Transformer model and make predictions.")
    parser.add_argument('--protein', type=str, default='DDX3X-HepG2_TGFB1', help='The protein dataset to use for training and prediction')
    args = parser.parse_args()

    protein = args.protein
    print(f"Using dataset: {protein}")

    trainX, trainX2 = dealwithdata(protein)

    model = U_Transformer()
    print(f"Loading previous best weights for {protein}")
    model.load_weights(f'../new2/Utransformer_weights_{protein}.h5')

    prediction = model.predict([trainX, trainX2])[:, 1]
    print(prediction, protein)

if __name__ == "__main__":
    main()
