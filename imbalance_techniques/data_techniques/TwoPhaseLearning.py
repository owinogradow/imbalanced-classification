import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

from imbalance_techniques.BalancingTechnique import BalancingTechnique
from imbalancer.DataProcessor import DataProcessor
from logger.Logger import logger

class TwoPhaseLearning(BalancingTechnique):
    def __init__(self, name):
        super(TwoPhaseLearning, self).__init__(name)

    def balanceDataset(self, dataset, labels):
        classes, classesCount = np.unique(labels, return_counts=True)
        highestClassCount = np.amax(classesCount)
        treshold = highestClassCount / 2
        for classId, classCount in zip(classes, classesCount):
            if classCount > treshold:
                amountToRemove = int(classCount - treshold)
                classIndexes = np.where(labels == classId)[0].tolist()
                indexesToRemove = random.sample(classIndexes, amountToRemove)
                dataset = np.delete(dataset, indexesToRemove, 0)
                labels = np.delete(labels, indexesToRemove, 0)
        return dataset, labels

    def handleImbalanceProblem(self, dataset, labels, model, epochs):
        model.getCompiledModel()
        model = model.model

        train_images, test_images, train_labels, test_labels = DataProcessor.trainTestSplit(dataset, labels)
        train_images, train_labels = self.balanceDataset(train_images, train_labels)
        train_images, test_images, train_labels, test_labels = DataProcessor.prepareDatasetForFit(train_images, test_images, train_labels, test_labels)
        
        history = model.fit(train_images, train_labels, epochs=epochs/2)
        

        train_images, test_images, train_labels, test_labels = DataProcessor.trainTestSplit(dataset, labels)
        train_images, test_images, train_labels, test_labels = DataProcessor.prepareDatasetForFit(train_images, test_images, train_labels, test_labels)
        
        history = model.fit(train_images, train_labels, epochs=epochs/2, validation_data=(test_images, test_labels)) # second phase

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
        