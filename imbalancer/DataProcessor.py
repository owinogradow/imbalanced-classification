import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import random

class DataProcessor:

    @staticmethod
    def trainTestSplit(dataset, labels):
        train_images, test_images, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.20, random_state=42)
        return train_images, test_images, train_labels, test_labels


    @staticmethod
    def prepareDatasetForFit(train_images, test_images, train_labels, test_labels):
        classes, classesCount = np.unique(np.concatenate((train_labels, test_labels)), return_counts=True)
        if len(classesCount) > 2:
            train_images, test_images = train_images / 255.0, test_images / 255.0
            train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(classesCount))
            test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(classesCount))
            
        return train_images, test_images, train_labels, test_labels

    @staticmethod
    def reduceDatasetSize(dataset, labels, reduceFactor):
        classes, classesCount = np.unique(labels, return_counts=True)
        for classId, classCount in zip(classes, classesCount):
            amountToRemove = int(classCount - classCount / reduceFactor)
            classIndexes = np.where(labels == classId)[0].tolist()
            indexesToRemove = random.sample(classIndexes, amountToRemove)
            dataset = np.delete(dataset, indexesToRemove, 0)
            labels = np.delete(labels, indexesToRemove, 0)
        return dataset, labels