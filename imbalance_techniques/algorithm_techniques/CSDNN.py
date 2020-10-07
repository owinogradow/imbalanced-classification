import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils import class_weight

from imbalance_techniques.BalancingTechnique import BalancingTechnique
from imbalancer.DataProcessor import DataProcessor
from logger.Logger import logger

class CSDNN(BalancingTechnique):

    def __init__(self, name):
        super(CSDNN, self).__init__(name)

    def handleImbalanceProblem(self, dataset, labels, model, epochs):
        model.getCompiledModel()
        model = model.model
        
        flatten = labels.flatten()
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=flatten)
        classWeightsMap = {idx : ratio for idx, ratio in enumerate(class_weights)}

        train_images, test_images, train_labels, test_labels = DataProcessor.trainTestSplit(dataset, labels)
        train_images, test_images, train_labels, test_labels = DataProcessor.prepareDatasetForFit(train_images, test_images, train_labels, test_labels)

        history =  model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels), class_weight=classWeightsMap)

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