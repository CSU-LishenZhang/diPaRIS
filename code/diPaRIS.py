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
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
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
        j = seqlen if i+win>seqlen else i+win
        yield seq[i:j]
        if j==seqlen: break
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
    seqP =[]
    seqN =[]
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
        dataX2.append(icshapeTrend(seqP[i], icshapeP[i]))

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
        dataX2.append(icshapeTrend(seqN[i], icshapeN[i]))

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

def SE(input, kernel_size):
  x = keras.layers.Conv2D(filters=64, kernel_size=kernel_size, padding="same")(input)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
  y = keras.backend.squeeze(x, axis=1)
  y = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(y)
  z = keras.layers.multiply([x, y])
  return z

def DowmSample(input, filters):
  x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same")(input)
  x = keras.layers.LayerNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same")(x)
  x = keras.layers.LayerNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  y = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(x)
  y = keras.layers.Dropout(0.3)(y)
  return x, y

def JumpConnect(input, filters):
  x = keras.backend.squeeze(input, axis=1)
  y = keras_nlp.layers.TransformerEncoder(intermediate_dim=filters, num_heads=filters, dropout=0.3)(x)
  y = keras_nlp.layers.TransformerEncoder(intermediate_dim=filters, num_heads=filters, dropout=0.3)(y)
  y = keras_nlp.layers.TransformerEncoder(intermediate_dim=filters, num_heads=filters, dropout=0.3)(y)
  z = keras.layers.multiply([x, y])

def UpSample(input1, input2, filters, padding):
  x = keras.layers.Conv1DTranspose(filters=filters, kernel_size=3, strides=2, padding=padding)(input1)
  x = keras.layers.LayerNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  y = Concatenate(axis=-1)([x, input2])
  y = keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same")(y)
  y = keras.layers.LayerNormalization()(y)
  y = keras.layers.Activation('relu')(y)
  y = keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same")(y)
  y = keras.layers.BatchNormalization()(y)
  y = keras.layers.Activation('relu')(y)
  return y
  
def U_Transformer():
    #alignment
    left_input = keras.layers.Input(shape=(1, 101, 4), name='left_input')
    right_input = keras.layers.Input(shape=(1, 101, 4), name='right_input')
    left_SE = SE(left_input, (6, 4))
    right_SE = SE(right_input, (10, 7))
    merge = keras.layers.Concatenate(axis=-2)([left_SE, right_SE])
    #reconstruction
    ##down-sampling
    D1, D2 = DowmSample(merge, 32)
    D3, D4 = DowmSample(D2, 64)
    D5, D6 = DowmSample(D4, 128)
    ##Jump-connection
    J1 = JumpConnect(D1, 32)
    J3 = JumpConnect(D3, 64)
    J5 = JumpConnect(D5, 128)
    ##Bottleneck
    B1 = keras.layers.Conv2D(256, (3, 3), padding="same")(D6)
    B1 = keras.layers.LayerNormalization()(B1)
    B1 = keras.layers.Activation('relu')(B1)
    B1 = keras.layers.Conv2D(256, (3, 3), padding="same")(B1)
    B1 = keras.layers.BatchNormalization()(B1)
    B1 = keras.layers.Activation('relu')(B1)
    B2 = keras.backend.squeeze(B1, axis=1)
    B2 = keras.layers.Conv1D(filters=256, kernel_size=3, padding="same")(B2)
    B2 = keras.layers.LayerNormalization()(B2)
    B2 = keras.layers.Activation('relu')(B2)
    B2 = keras.layers.Conv1D(filters=256, kernel_size=3, padding="same")(B2)
    B2 = keras.layers.BatchNormalization()(B2)
    B2 = keras.layers.Activation('relu')(B2)
    #up-sampling
    U1 = UpSample(B2, J5, 128, "same")
    U2 = UpSample(U1, J3, 64, "valid")
    U3 = UpSample(U2, J1, 32, "same")
    #classify
    stack1 = keras.layers.LayerNormalization()(U3)
    stack2 = keras.layers.AveragePooling1D(pool_size=int(stack1.shape[1]))(stack1)
    stack3 = keras.layers.AveragePooling1D(40)(stack1)
    stack4 = keras.layers.AveragePooling1D(8)(stack1)
    stack6 = Concatenate(axis=1)([stack2, stack3, stack4])
    stack7 = GlobalExpectationPooling1D(mode=0, m_trainable=False, m_value=1)(stack6)
    output = keras.layers.Dense(2, activation="softmax")(stack7)
    return Model(inputs=[left_input, right_input], outputs=[output])

def main():
    protein_list = {0: 'AKAP1-HepG2'}
    for p in range(0, 1, 1):
        protein = protein_list[p]
        print(protein)
        trainXeval, test_X, trainYeval, test_y = dealwithdata(protein)
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
            
            model = U_Transformer()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            if os.path.exists('diPaRIS_' + '_' + protein + '.h5'):
                print("load previous best weights for model")
                model.load_weights('diPaRIS_' + '_' + protein + '.h5')
              
            def step_decay(epoch):
                initial_lrate = 0.0005
                drop = 0.8
                epochs_drop = 5.0
                lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                print(lrate)
                return lrate
              
            callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'), LearningRateScheduler(step_decay)]  
            history = model.fit(train_X, train_y, batch_size=16, epochs=64, verbose=0, validation_data=(eval_X, eval_y), callbacks=callbacks)
            model.save('diPaRIS_' + '_' + protein + '.h5')
            
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
        
       print("AUC: %.4f %.4f %.4f %.4f %.4f" % (auc_list[0], auc_list[1], auc_list[2], auc_list[3], auc_list[4]), protein)
       print("ACC: %.4f %.4f %.4f %.4f %.4f" % (acc_list[0], acc_list[1], acc_list[2], acc_list[3], acc_list[4]), protein)
       print("Precision: %.4f %.4f %.4f %.4f %.4f" % (precision_list[0], precision_list[1], precision_list[2], precision_list[3], precision_list[4]), protein)
       print("Recall: %.4f %.4f %.4f %.4f %.4f" % (recall_list[0], recall_list[1], recall_list[2], recall_list[3], recall_list[4]), protein)
       print("F1-score: %.4f %.4f %.4f %.4f %.4f" % (f1_score_list[0], f1_score_list[1], f1_score_list[2], f1_score_list[3], f1_score_list[4]), protein)
       print("AUPR: %.4f %.4f %.4f %.4f %.4f" % (aupr_list[0], aupr_list[1], aupr_list[2], aupr_list[3], aupr_list[4]), protein)

if __name__ == "__main__":
    main()
