import numpy as np
import json
import os
import pickle
import random
import sys
import yaml
import numpy as np
import tensorflow as tf
from callbacks  import make_callbacks
from datasets   import make_seq_2_seq_dataset, Seq2SeqDataset, Seq2SeqDataset_copy
from metrics    import get_metrics
from models     import create_model
from outputs    import plot_metrics, plot_predictions, save_model
from params     import get_params
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, RepeatVector, Reshape, TimeDistributed

def main():
    params=get_params()

    datasetD = make_seq_2_seq_dataset(params)
    train_x = datasetD['train']['x']
    train_y = datasetD['train']['y']
    test_x = datasetD['test']['x']
    test_y = datasetD['test']['y']
    train_scenarios = datasetD['train']['scenarios']
    test_scenarios = datasetD['test']['scenarios']
    val_scenarios = datasetD['val']['scenarios']
    params.scaleD = datasetD['scaleD']


    input_window_samps = params.input_window_length_samples
    num_signals = params.num_signals
    output_window_samps = params.output_window_length_samples



    train_X=np.array(train_x,dtype=float)
    # train_X=np.reshape(train_X,(input_window_samps,-1))

    #
    train_Y=np.array(train_y,dtype=float)
    nsamples, nx, ny = train_Y.shape
    # print(nsamples)
    # print(nx)
    # print(ny)

    train_Y=np.reshape(train_Y,(nsamples,output_window_samps*num_signals))
    print(train_X.shape)
    #print(len(train_X[0,:]))
    print(train_Y.shape)

    print(train_Y)







    # print(input_window_samps)
    # print(num_signals)
    # print(output_window_samps)
    # train_x = Reshape((input_window_samps,num_signals))(train_x)
    # print(train_x.shape)
    # print(train_x[0,:])
    #
    # train_y = Reshape((output_window_samps, num_signals))(train_y)
    # print(train_y.shape)
    # print(train_y[0,:])

    model = SVR(kernel='rbf', degree=3)
    history = model.fit(train_X, train_Y)
    history_score = history.score(test_x, test_y)
    print("the score of linear is : %f" % history_score)




if __name__=="__main__":
    main()

