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
    
    for seq, struc in zip(seq_list, struc_list):  # 遍历多行输入
        dataX.append(coden(seq))
        dataX2.append(icshapeDS(seq, struc))
    
    dataX = np.array(dataX)
    dataX = dataX[:, np.newaxis, :] 
    dataX2 = np.array(dataX2)
    dataX2 = dataX2[:, np.newaxis, :] 

    return dataX, dataX2

def diPaRIS():
    # input
    left_input = keras.layers.Input(shape=(1, 101, 4), name='left_input')
    right_input = keras.layers.Input(shape=(1, 100, 7), name='right_input')
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

protein_list = {
    0: 'AKAP1-HepG2', 1: 'AQR-HepG2', 2: 'AQR-K562', 3: 'BCLAF1-HepG2', 4: 'BUD13-HepG2',
    5: 'BUD13-K562', 6: 'DDX24-K562', 7: 'DDX3X-HepG2', 8: 'DDX3X-K562', 9: 'EFTUD2-HepG2',
    10: 'EFTUD2-K562', 11: 'FAM120A-K562', 12: 'FMR1-K562', 13: 'FXR2-K562', 14: 'G3BP1-HepG2',
    15: 'GRWD1-HepG2', 16: 'GRWD1-K562', 17: 'IGF2BP1-K562', 18: 'IGF2BP2-K562', 19: 'LARP4-HepG2',
    20: 'LIN28B-K562', 21: 'METAP2-K562', 22: 'PABPC4-K562', 23: 'PABPN1-HepG2', 24: 'PCBP2-HepG2',
    25: 'PPIG-HepG2', 26: 'PRPF8-HepG2', 27: 'PRPF8-K562', 28: 'PUM1-K562', 29: 'PUM2-K562',
    30: 'RBM15-K562', 31: 'RPS3-HepG2', 32: 'RPS3-K562', 33: 'SF3B4-HepG2', 34: 'SF3B4-K562',
    35: 'SND1-HepG2', 36: 'SND1-K562', 37: 'SUB1-HepG2', 38: 'UCHL5-K562', 39: 'UPF1-HepG2',
    40: 'UPF1-K562', 41: 'YBX3-K562', 42: 'ZNF622-K562', 43: 'ZNF800-K562'
}

def predict_with_model(model, seq, struc, protein_name):
    dataX, dataX2 = dealwithdata(seq, struc)
    prediction = model.predict([dataX, dataX2])[:, 1]
    print(f"Prediction for {protein_name}: {prediction}")
    return prediction

def main(seq, struc, protein=None):
    if protein:
        # 预测指定数据集
        if protein in protein_list.values():
            print(f"Using model trained on {protein}")
            model = diPaRIS()
            model.load_weights(f'../model/diPaRIS_{protein}.h5')# 此处应加载指定数据集的预训练模型
            return predict_with_model(model, seq, struc, protein)
        else:
            print(f"Specified dataset '{protein}' is not in the list.")
    else:
        # 预测所有数据集
        for key, protein_name in protein_list.items():
            print(f"Using model trained on {protein_name}")
            model = diPaRIS()  
            model.load_weights(f'../model/diPaRIS_{protein_name}.h5')# 此处应加载每个数据集对应的预训练模型
            predict_with_model(model, seq, struc, protein_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='diPaRIS Prediction')
    parser.add_argument('--seq', type=str, required=True, help='Input sequence')
    parser.add_argument('--struc', type=float, nargs='+', required=True, help='Input structure data')
    parser.add_argument('--protein', type=str, help='Protein dataset to use (optional)')
    args = parser.parse_args()

    main(args.seq, args.struc, args.protein)
