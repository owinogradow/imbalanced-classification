import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

from imbalance_techniques.BalancingTechnique import BalancingTechnique
from imbalancer.DataProcessor import DataProcessor
from logger.Logger import logger

class DatasetSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, test_data, test_labels, model, batch_size=32):
        self.train_data, self.train_labels = x_set, y_set
        self.test_data, self.test_labels = test_data, test_labels
        self.batch_size = batch_size
        self.model = model
        self.hardMiningSample = 500
        self.counter=0

    def __len__(self):
        return int(np.ceil(len(self.train_data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.train_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.train_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.counter += 1
        if self.counter == 10:
            logger.info("return")
            return

        predictions = self.model.predict(self.train_data)
        loss = tfa.losses.sigmoid_focal_crossentropy(tf.cast(self.train_labels, tf.float32), predictions)

        if predictions.shape[1] > 1:
            y_true = np.argmax(self.train_labels, axis=1)
            y_pred = np.argmax(predictions, axis=1)
        else:
            y_true = self.train_labels
            y_pred = np.where(predictions > 0.5, 1, 0)
            
        classes, classesCount = np.unique(y_true, return_counts=True)
        highestClassCount = np.amax(classesCount)

        for classId, classCount in zip(classes, classesCount):
            
            if classCount < highestClassCount:
                # hard mining
                amountToCopy = highestClassCount - classCount
                classIndexes = np.where(y_true == classId)[0].tolist()
                classesLoss = [ ( idx, loss[idx] ) for idx in classIndexes]
                classesLoss.sort(key=lambda x : x[1])
                indexesToCopy = [ x[0] for x in classesLoss[-self.hardMiningSample:]]

                for i in range(int(amountToCopy/self.hardMiningSample)):
                    copiedData = np.copy(self.train_data[indexesToCopy,])
                    copiedLabels = np.copy(self.train_labels[indexesToCopy,])
                    self.train_data = np.concatenate((self.train_data, copiedData))
                    self.train_labels = np.concatenate((self.train_labels, copiedLabels))
        
        logger.info(np.unique(self.train_data)[1])
        logger.info(self.train_data.shape)
        logger.info(self.train_labels.shape)

class CRLHardMining(BalancingTechnique):

    def __init__(self, name):
        super(CRLHardMining, self).__init__(name)

    def handleImbalanceProblem(self, dataset, labels, model, epochs):
        model.getCompiledModel(loss=tfa.losses.SigmoidFocalCrossEntropy(False))
        model = model.model
        
        train_images, test_images, train_labels, test_labels = DataProcessor.trainTestSplit(dataset, labels)
        train_images, test_images, train_labels, test_labels = DataProcessor.prepareDatasetForFit(train_images, test_images, train_labels, test_labels)

        dataSequence = DatasetSequence(train_images, train_labels, test_images, test_labels, model)
        history =  model.fit(dataSequence, epochs=epochs, validation_data=(test_images, test_labels))

        predictions = model.predict(dataSequence.test_data)

        if predictions.shape[1] > 1:
            test_labels_orig = np.argmax(dataSequence.test_labels, axis=1)
            test_labels_pred = np.argmax(predictions, axis=1)
        else:
            test_labels_orig = dataSequence.test_labels
            test_labels_pred = np.where(predictions > 0.5, 1, 0)
        logger.info(classification_report(y_true=test_labels_orig, y_pred=test_labels_pred))

        logger.info(precision_recall_fscore_support(y_true=test_labels_orig, y_pred=test_labels_pred, average='weighted'))
        return history