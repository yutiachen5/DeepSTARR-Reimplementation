# adapted from Deepstarr colab notebook: https://colab.research.google.com/drive/1Xgak40TuxWWLh5P5ARf0-4Xo0BcRn0Gd 

import sys
import time
import sklearn
import json
import tensorflow as tf
import keras
import keras_nlp
import keras.layers as kl
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder
from keras.layers import BatchNormalization
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
from itertools import product
from sklearn.metrics import mean_squared_error

startTime=time.time()

def LoadConfig(config):
    with open(config, 'r') as file:
        params = json.load(file)
    return params

def one_hot_encode(seq):
    nucleotide_dict = {'A': [1, 0, 0, 0],
                       'C': [0, 1, 0, 0],
                       'G': [0, 0, 1, 0],
                       'T': [0, 0, 0, 1],
                       'N': [0, 0, 0, 0]} 
    return np.array([nucleotide_dict[nuc] for nuc in seq])

def kmer_encode(sequence, k=3):
    sequence = sequence.upper()
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)
    return {kmer: kmer_counts.get(kmer, 0) / len(kmers) for kmer in [''.join(p) for p in product('ACGT', repeat=k)]}

def kmer_features(seq, k=3):
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    feature_matrix = []
    kmer_freqs = kmer_encode(seq, k)
    feature_vector = [kmer_freqs[kmer] for kmer in all_kmers]
    feature_matrix.append(feature_vector)
    return np.array(feature_matrix)

def prepare_input(data_set, params):
    if params['encode'] == 'one-hot':
        seq_matrix = np.array(data_set['Sequence'].apply(one_hot_encode).tolist())  # (number of sequences, length of sequences, nucleotides)
    elif params['encode'] == 'k-mer':
        seq_matrix = np.array(data_set['Sequence'].apply(kmer_features, k=3).tolist())  # (number of sequences, 1, 4^k)
    else:
        raise Exception ('wrong encoding method')

    Y_dev = data_set.Dev_log2_enrichment
    Y_hk = data_set.Hk_log2_enrichment
    Y = [Y_dev, Y_hk]

    return seq_matrix, Y

def DeepSTARR(params):
    if params['encode'] == 'one-hot':
        input = kl.Input(shape=(249, 4)) 
    elif params['encode'] == 'k-mer':
        input = kl.Input(shape=(1, 64)) 

    for i in range(params['convolution_layers']['n_layers']):
        x = kl.Conv1D(params['convolution_layers']['filters'][i],
                      kernel_size = params['convolution_layers']['kernel_sizes'][i],
                      padding = params['pad'],
                      name=str('Conv1D_'+str(i+1)))(input)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        if params['encode'] == 'one-hot':
            x = kl.MaxPooling1D(2)(x)

        if params['dropout_conv'] == 'yes': x = kl.Dropout(params['dropout_prob'])(x)

    # optional attention layers
    for i in range(params['transformer_layers']['n_layers']):
        if i == 0:
            x = x + keras_nlp.layers.SinePositionEncoding()(x)
        x = TransformerEncoder(intermediate_dim = params['transformer_layers']['attn_key_dim'][i],
                                num_heads = params['transformer_layers']['attn_heads'][i],
                                dropout = params['dropout_prob'])(x)
    
    # After the convolutional layers, the output is flattened and passed through a series of fully connected/dense layers
    # Flattening converts a multi-dimensional input (from the convolutions) into a one-dimensional array (to be connected with the fully connected layers
    x = kl.Flatten()(x)
    
    # Fully connected layers
    # Each fully connected layer is followed by batch normalization, ReLU activation, and dropout
    for i in range(params['n_dense_layer']):
        x = kl.Dense(params['dense_neurons'+str(i+1)],
                     name=str('Dense_'+str(i+1)))(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(params['dropout_prob'])(x)
    
    # Main model bottleneck
    bottleneck = x
    
    # heads per task (developmental and housekeeping enhancer activities)
    # The final output layer is a pair of dense layers, one for each task (developmental and housekeeping enhancer activities), each with a single neuron and a linear activation function
    tasks = ['Dev', 'Hk']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))
    
    # Build Keras model object
    model = Model([input], outputs)
    model.compile(Adam(learning_rate=params['lr']), # Adam optimizer
                  loss=['mse', 'mse'], # loss is Mean Squared Error (MSE)
                  loss_weights=[1, 1]) # in case we want to change the weights of each output. For now keep them with same weights

    return model, params

def train(selected_model, X_train, Y_train, X_valid, Y_valid, params):
    my_history=selected_model.fit(X_train, Y_train,
                                  validation_data=(X_valid, Y_valid), 
                                  batch_size=params['batch_size'],
                                  epochs=params['epochs'],
                                  callbacks=[EarlyStopping(patience=params['early_stop'], monitor="val_loss", restore_best_weights=True), History()])

    return selected_model, my_history

def summary_statistics(X, Y, set, task, main_model, main_params):
    pred = main_model.predict(X, batch_size=main_params['batch_size']) # predict
    if task =="Dev":
        i=0
    if task =="Hk":
        i=1
    print(set + ' MSE ' + task + ' = ' + str("{0:0.2f}".format(mean_squared_error(Y, pred[i].squeeze()))))
    print(set + ' PCC ' + task + ' = ' + str("{0:0.2f}".format(stats.pearsonr(Y, pred[i].squeeze())[0])))
    print(set + ' SCC ' + task + ' = ' + str("{0:0.2f}".format(stats.spearmanr(Y, pred[i].squeeze())[0]))) 

def my_scatter_density(X, Y, set, task, main_model, main_params, label):
    pred = main_model.predict(X, batch_size=main_params['batch_size'])
    if task =="Dev":
        i=0
        c="red"
    if task =="Hk":
        i=1
        c="blue"

    g = sns.jointplot(x=Y, y=pred[i].squeeze(), kind="kde", fill=True, color=c)
    g.ax_marg_x.remove() # remove marginal densities
    g.ax_marg_y.remove() # remove marginal densities

    # add regression line of data
    # sns.regplot(x=Y, y=pred[i].squeeze(), scatter=False, color='black', ax=g.ax_joint)

    # add expected regression line
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, 'w', linestyle='dashed', transform=g.ax_joint.transData, color='grey')

    # same axes ranges
    g.ax_joint.set_aspect('equal')

    PCC = str("{0:0.2f}".format(stats.pearsonr(Y, pred[i].squeeze())[0]))
    plt.xlabel('Measured expression [log2]')
    plt.ylabel('Predicted expression [log2]')
    plt.title(str(task + ' - ' + set + ' set (PCC=' + PCC + ')'))
    plt.savefig(outdir + '/' + label + '-' + set + '-' + task + '.png')


def main(config, indir, outdir, label):
    data = pd.read_table(indir)
    params = LoadConfig(config)

    X_train, Y_train = prepare_input(data[data['set'] == "Train"], params)
    X_valid, Y_valid = prepare_input(data[data['set'] == "Val"], params)
    X_test, Y_test = prepare_input(data[data['set'] == "Test"], params)

    DeepSTARR(params)[0].summary() 
    DeepSTARR(params)[1] 
    main_model, main_params = DeepSTARR(params)
    main_model, my_history = train(main_model, X_train, Y_train, X_valid, Y_valid, main_params)

    endTime=time.time()
    seconds=endTime-startTime
    print("Total training time:",round(seconds/60,2),"minutes")

    summary_statistics(X_train, Y_train[0], "train", "Dev", main_model, main_params)
    summary_statistics(X_valid, Y_valid[0], "validation", "Dev", main_model, main_params)
    summary_statistics(X_test, Y_test[0], "test", "Dev", main_model, main_params)

    summary_statistics(X_train, Y_train[1], "train", "Hk", main_model, main_params)
    summary_statistics(X_valid, Y_valid[1], "validation", "Hk", main_model, main_params)
    summary_statistics(X_test, Y_test[1], "test", "Hk", main_model, main_params)

    my_scatter_density(X_test, Y_test[0], "test", "Dev", main_model, main_params, label)
    my_scatter_density(X_test, Y_test[1], "test", "Hk", main_model, main_params, label)

    main_model.save(outdir + '/' + label + '.h5')

if (len(sys.argv)!=5):
    exit("<parms.config> <data-indir> <outdir> <out-label>\n")

(config, indir, outdir, label)=sys.argv[1:]
main(config, indir, outdir, label)


