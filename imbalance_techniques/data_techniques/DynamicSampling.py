import numpy as np
import random
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

from logger.Logger import logger
from imbalance_techniques.BalancingTechnique import BalancingTechnique
from imbalancer.DataProcessor import DataProcessor
from imbalancer.DataImbalancer import DataImbalancer
from imbalance_techniques.data_techniques.RandomOverSampler import RandomOverSampler
class DatasetSequence(tf.keras.utils.Sequence):
    
    def __init__(self, x_set, y_set, test_data, test_labels, model, epochs, batch_size=32):
        self.train_data, self.train_labels = x_set, y_set
        self.test_data, self.test_labels = test_data, test_labels
        self.batch_size = batch_size
        self.model = model
        self.epochs = epochs
        self.counter=0

    def __len__(self):
        return int(np.ceil(len(self.train_data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.train_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.train_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):

        self.counter += 1
        if self.counter == self.epochs:
            logger.info("return")
            return
        predictions = self.model.predict(self.train_data)
        y_true = None
        y_pred = None
        if self.train_labels.shape[1] > 1:
            y_true = np.argmax(self.train_labels, axis=1)
            y_pred = np.argmax(predictions, axis=1)
        else:
            y_true = self.train_labels
            y_pred = np.where(predictions > 0.5, 1, 0)

        report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        if self.counter == self.epochs -1:
            logger.info(classification_report(y_true=y_true, y_pred=y_pred))
        
        classesCounts = np.unique(y_true, return_counts=True)[1]
        averageClassSize = math.ceil(np.mean(classesCounts))

        totalsScores = [ report[label]['f1-score'] for label in report if label not in ["accuracy", "macro avg", "weighted avg"]]
        for ix, s in enumerate(totalsScores):
            logger.info("{} : {}".format(ix, s))

        totals = [ 1 - report[label]['f1-score'] for label in report if label not in ["accuracy", "macro avg", "weighted avg"]]
        f1ScoreTotal = np.sum( [ 1 - report[label]['f1-score'] for label in report if label not in ["accuracy", "macro avg", "weighted avg"]] ) / len(classesCounts)
        updatedSampleSize = {}
        for label in report:
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                updatedSampleSize[int(label)] = int(( (1 - report[label]['f1-score']) / f1ScoreTotal ) * averageClassSize)
        logger.info(updatedSampleSize)
        logger.info("mean class size: {}".format(averageClassSize))
        classes, classesCount = np.unique(y_true, return_counts=True)
        logger.info(classesCount)

        for classId, classCount in zip(classes, classesCount):
            if classCount > updatedSampleSize[classId]:
                logger.info("Undersample class {}".format(classId))
                amountToRemove = classCount - updatedSampleSize[classId]
                classIndexes = np.where(y_true == classId)[0].tolist()
                indexesToRemove = random.sample(classIndexes, amountToRemove)
                self.train_data = np.delete(self.train_data, indexesToRemove, 0)
                self.train_labels = np.delete(self.train_labels, indexesToRemove, 0)

            elif classCount < updatedSampleSize[classId]:
                logger.info("Oversample class {}".format(classId))
                amountToCopy = updatedSampleSize[classId] - classCount
                classIndexes = np.where(y_true == classId)[0].tolist()
                indexesToCopy = random.choices(classIndexes, k=amountToCopy)
                copiedDataset = np.copy(self.train_data[indexesToCopy,])
                copiedLabels = np.copy(self.train_labels[indexesToCopy,])
                self.train_data = np.concatenate((self.train_data, copiedDataset))
                self.train_labels = np.concatenate((self.train_labels, copiedLabels))
                
            if self.train_labels.shape[1] > 1:
                y_true = np.argmax(self.train_labels, axis=1)
            else:
                y_true = self.train_labels

class DynamicSampling(BalancingTechnique):
    
    def __init__(self, name):
        super(DynamicSampling, self).__init__(name)

    def handleImbalanceProblem(self, dataset, labels, model, epochs):
        model.getCompiledModel()
        model = model.model

        test_labels_orig = test_labels_pred = None

        train_images, test_images, train_labels, test_labels = DataProcessor.trainTestSplit(dataset, labels)
        train_images, test_images, train_labels, test_labels = DataProcessor.prepareDatasetForFit(train_images, test_images, train_labels, test_labels)

        dataSequence = DatasetSequence(train_images, train_labels, test_images, test_labels, model, epochs)
        history = model.fit(dataSequence, epochs=epochs, validation_data=(test_images, test_labels))

        predictions = model.predict(dataSequence.test_data)
        if predictions.shape[1] > 1:
            test_labels_orig = np.argmax(dataSequence.test_labels, axis=1)
            test_labels_pred = np.argmax(predictions, axis=1)
        else:
            test_labels_orig = dataSequence.test_labels
            test_labels_pred = np.where(predictions > 0.5, 1, 0)
        logger.info(classification_report(y_true=test_labels_orig, y_pred=test_labels_pred, zero_division=1))
        logger.info(precision_recall_fscore_support(y_true=test_labels_orig, y_pred=test_labels_pred, average='weighted'))
        return history
