from tensorflow.keras import layers, models, metrics, optimizers, losses

class RNN:

    def __init__(self):
        self.model = models.Sequential([
                layers.Embedding(20000, 64),
                layers.Bidirectional(layers.LSTM(64)),
                layers.Dense(64, activation='relu'),
                layers.Dense(1)
            ])
    
    def getCompiledModel(self, optimizer=None, loss=None, runEagerly=False):
        self.model.compile(
            optimizer=optimizers.Adam(1e-4) if optimizer is None else optimizer,
            loss=losses.BinaryCrossentropy(from_logits=True) if loss is None else loss,
            metrics=['accuracy'],
            run_eagerly=runEagerly)