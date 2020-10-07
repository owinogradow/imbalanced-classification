import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import keras.backend as K

from imbalance_techniques.BalancingTechnique import BalancingTechnique
from imbalancer.DataProcessor import DataProcessor
from logger.Logger import logger

class MSFE(BalancingTechnique):

    def __init__(self, name):
        super(MSFE, self).__init__(name)

    def argmax(self, x, beta=1e10):
        x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
        return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

    def meanFalseError(self, y_true, y_pred):
        neg_y_pred = 1 - y_pred
        neg_y_true = 1 - y_true

        fp = K.sum(neg_y_true * y_pred)
        fn = K.sum(y_true * neg_y_pred)
        fps = K.square(K.sum(fp))
        fns = K.square(K.sum(fn))
        summ = K.sum(fps+fns)
        
        return summ
        
    def handleImbalanceProblem(self, dataset, labels, model, epochs):
        model.getCompiledModel(loss=self.meanFalseError, runEagerly=True)
        model = model.model
        
        train_images, test_images, train_labels, test_labels = DataProcessor.trainTestSplit(dataset, labels)
        train_images, test_images, train_labels, test_labels = DataProcessor.prepareDatasetForFit(train_images, test_images, train_labels, test_labels)
        
        history =  model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

        predictions = model.predict(test_images)
        if predictions.shape[1] > 1:
            test_labels_orig = np.argmax(test_labels, axis=1)
            test_labels_pred = np.argmax(predictions, axis=1)
        else:
            test_labels_orig = test_labels
            test_labels_pred = np.where(predictions > 0.5, 1, 0)
        logger.info(classification_report(y_true=test_labels_orig, y_pred=test_labels_pred))

        logger.info(precision_recall_fscore_support(y_true=test_labels_orig, y_pred=test_labels_pred, average='weighted'))
        return history