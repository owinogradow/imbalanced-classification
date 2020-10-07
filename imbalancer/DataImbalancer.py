import numpy as np
import random

class DataImbalancer:

    @staticmethod
    def imbalanceImageDataset(dataset, labels, imbalanceRatio, labelsToImbalance):
        classes, classesCount = np.unique(labels, return_counts=True)
        indexesToRemove = []
        for label in labelsToImbalance:
            classLabels = np.where(labels == label)[0].tolist()
            labelsToRemoveLen = int(len(classLabels) * (1 - imbalanceRatio))
            indexesToRemove.extend(random.sample(classLabels, labelsToRemoveLen))
        indexesToRemove.sort()
        dataset = np.delete(dataset, indexesToRemove, 0)
        labels = np.delete(labels, indexesToRemove, 0)
        if len(dataset) != len(labels):
            raise RuntimeError("Imbalancing went wrong, dataset and labels have different size!")
        return dataset, labels