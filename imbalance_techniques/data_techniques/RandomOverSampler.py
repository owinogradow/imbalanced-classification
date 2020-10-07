import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

from imbalance_techniques.BalancingTechnique import BalancingTechnique
from imbalancer.DataProcessor import DataProcessor
from logger.Logger import logger

class RandomOverSampler(BalancingTechnique):
    
    def __init__(self, name):
        super(RandomOverSampler, self).__init__(name)

    def balanceDataset(self, dataset, labels):
        classes, classesCount = np.unique(labels, return_counts=True)
        logger.info(classesCount)
        highestClassCount = np.amax(classesCount)
        for classId, classCount in zip(classes, classesCount):
            if classCount < highestClassCount:
                amountToCopy = highestClassCount - classCount
                logger.info("Oversampling class {} with {} samples".format(classId, amountToCopy))
                classIndexes = np.where(labels == classId)[0].tolist()
                indexesToCopy = random.choices(classIndexes, k=amountToCopy)
                copiedDataset = np.copy(dataset[indexesToCopy,])
                copiedLabels = np.copy(labels[indexesToCopy,])

                dataset = np.concatenate((dataset, copiedDataset))
                labels = np.concatenate((labels, copiedLabels))
        return dataset, labels

    def handleImbalanceProblem(self, dataset, labels, model, epochs):
        model.getCompiledModel()
        model = model.model
        
        train_images, test_images, train_labels, test_labels = DataProcessor.trainTestSplit(dataset, labels)
        
        train_images, train_labels = self.balanceDataset(train_images, train_labels)

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