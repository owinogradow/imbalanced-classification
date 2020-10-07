from imbalance_techniques.data_techniques.RandomUnderSampler import RandomUnderSampler
from imbalance_techniques.data_techniques.RandomOverSampler import RandomOverSampler
from imbalance_techniques.data_techniques.TwoPhaseLearning import TwoPhaseLearning
from imbalance_techniques.data_techniques.DynamicSampling import DynamicSampling

from imbalance_techniques.algorithm_techniques.FocalLoss import FocalLoss
from imbalance_techniques.algorithm_techniques.MSFE import MSFE
from imbalance_techniques.algorithm_techniques.CSDNN import CSDNN

from imbalance_techniques.hybrid_techniques.CRLHardMining import CRLHardMining

from models.CNN import CNN
from models.RNN import RNN

class SimpleFactory:
    def __init__(self):
        pass

    def createHandleImbObj(self, handleImbMethod):
        if handleImbMethod == "rus":
            return RandomUnderSampler("random under sampling")
        elif handleImbMethod == "ros":
            return RandomOverSampler("random over sampling")
        elif handleImbMethod == "tpl":
            return TwoPhaseLearning("two-phase learning")
        elif handleImbMethod == "ds":
            return DynamicSampling("dynamic sampling")
        elif handleImbMethod == "fl":
            return FocalLoss("focal loss")
        elif handleImbMethod == "csdnn":
            return CSDNN("cost sensitive deep neural network")
        elif handleImbMethod == "mfse":
            return MSFE("mean false square error")
        elif handleImbMethod == "crlahm":
            return CRLHardMining("class rectification loss - hard mining")

    def createNeuralNetworkModel(self, neuralNetworkModel):
        if neuralNetworkModel == "cnn":
            return CNN()
        elif neuralNetworkModel == "rnn":
            return RNN()
