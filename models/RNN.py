import tensorflow as tf

class RNN:

    def __init__(self):
        self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(20000, 64),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
    
    def getCompiledModel(self, optimizer=None, loss=None, runEagerly=False):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4) if optimizer is None else optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True) if loss is None else loss,
            metrics=['accuracy'],
            run_eagerly=runEagerly)