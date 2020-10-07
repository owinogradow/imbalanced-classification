
from tensorflow.keras import layers, models, metrics
import tensorflow as tf
class CNN:
    def __init__(self):
      self.model = models.Sequential()
      self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
      self.model.add(layers.MaxPooling2D((2, 2)))
      self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      self.model.add(layers.MaxPooling2D((2, 2)))
      self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      self.model.add(layers.Flatten())
      self.model.add(layers.Dense(64, activation='relu'))
      self.model.add(layers.Dense(10, activation='softmax'))

    def getCompiledModel(self, optimizer=None, loss=None, runEagerly=False):
      self.model.compile(
          optimizer='adam' if optimizer is None else optimizer,
          loss='categorical_crossentropy' if loss is None else loss,
          metrics=[ 
            'accuracy', metrics.Recall(), metrics.Precision()
          ],
          run_eagerly=runEagerly)
