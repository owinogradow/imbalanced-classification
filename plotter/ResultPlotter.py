import matplotlib.pyplot as plt
from logger.Logger import logger

class ResultPlotter:
    @staticmethod
    def plotResults(history):
        try:
            plt.plot(history.history['accuracy'], label='accuracy')
            plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            # plt.ylim([0.5, 1])
            plt.legend(loc='lower right')
            plt.show()
        except KeyError as ke:
            logger.error("Caught key error: {}".format(ke.__str__()))
