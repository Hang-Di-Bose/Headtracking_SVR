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


# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt

def main():
    # tf.compat.v1.enable_v2_behavior()
    print("tensorflow version =",tf.__version__) 
    # get and save params of this run
    params = get_params()

    # dataset = Seq2SeqDataset_copy(
    #     input_path=params.input_dir,
    #     input_window_length_samples =params.input_window_length_samples,
    #     output_window_length_samples=params.output_window_length_samples,
    # )

    # train_dataset = tf.data.Dataset.from_generator((train_x, train_y),output_types=(tf.float64,tf.float64))
    # train_dataset = train_dataset.shuffle(buffer_size=100000)
    # train_dataset = train_dataset.repeat()
    
    datasetD = make_seq_2_seq_dataset(params)

    train_x = datasetD['train']['x']
    train_y = datasetD['train']['y']
    test_x  = datasetD['test']['x']
    test_y  = datasetD['test']['y']
    val_x   = datasetD['val']['x']
    val_y   = datasetD['val']['y']

    train_scenarios = datasetD['train']['scenarios']
    test_scenarios = datasetD['test']['scenarios']
    val_scenarios = datasetD['val']['scenarios']
    params.scaleD = datasetD['scaleD']  # store scaleD in params_out.yml
    
    #model = create_model(params)
    #model.compile(optimizer=params.optimizer,
    #              loss=params.loss,
     #             metrics=get_metrics(params))
    input_window_samps=params.input_window_length_samples
    num_signals = params.num_signals
    output_window_samps = params.output_window_length_samples



    train_x=tf.reshape(train_x,[input_window_samps,num_signals])
    train_y=tf.reshape(train_y,[output_window_samps,num_signals])
    test_x=tf.reshape(test_x,[input_window_samps,num_signals])
    test_y=tf.reshape(test_y,[output_window_samps,num_signals])
    model = SVR(kernel='rbf',degree=3)
    history=model.fit([train_x],[train_y])
    history_score=history.score(test_x,test_y)
    print("the score of linear is : %f" %history_score)







if __name__=="__main__":
    main()


