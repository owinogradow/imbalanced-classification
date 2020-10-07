import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import argparse
import sys

from models.CNN import CNN
from models.RNN import RNN
from logger.Logger import logger
from plotter.ResultPlotter import ResultPlotter
from imbalancer.DataImbalancer import DataImbalancer
from imbalancer.DataProcessor import DataProcessor
from factories.SimpleFactory import SimpleFactory

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser()
parser.add_argument('imbalance_ratio', type=float, help='level of imbalance, for instance 0.2, 0.05, 1 for original dataset')
parser.add_argument('model', 
    choices=['cnn', 'rnn'],
    help='neural network model used for classification, \navailable models: %(choices)s')
parser.add_argument('method',
    choices=['rus', 'ros', 'tpl', 'ds', 'fl', 'csdnn', 'mfse', 'crlahm'], 
    help='method to handle imbalance problem, \navailbale techniques: %(choices)s')

if __name__ == '__main__':
    args = parser.parse_args()
    
    factory = SimpleFactory()
    handleImbObj = factory.createHandleImbObj(args.method)
    neuralNetworkModel = factory.createNeuralNetworkModel(args.model)

    if isinstance(neuralNetworkModel, CNN):
        (train_data, train_labels), (test_data, test_labels) = datasets.cifar10.load_data()    # CIFAR-10
        concatenatedData = np.concatenate((train_data, test_data))
    
    elif isinstance(neuralNetworkModel, RNN):
        (train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words=20000)         # IMBD - REVIEWS
        concatenatedData = np.concatenate((train_data, test_data))
        concatenatedData = tf.keras.preprocessing.sequence.pad_sequences(concatenatedData, maxlen=80)
    
    try:
        data, labels = concatenatedData, np.concatenate((train_labels, test_labels))
        # data, labels = DataProcessor.reduceDatasetSize(concatenatedData, np.concatenate((train_labels, test_labels)), 2)
        data, labels = DataImbalancer.imbalanceImageDataset(data, labels, args.imbalance_ratio, [0])
    except RuntimeError as re:
        logger.error('Imbalance error {}'.format(re.__str__()))

    history = handleImbObj.handleImbalanceProblem(data, labels, neuralNetworkModel, epochs=10)
    ResultPlotter.plotResults(history)
