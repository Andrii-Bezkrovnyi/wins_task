from abc import ABC


class MnistClassifierInterface(ABC):
    """Abstract base class that defines the interface for all MNIST classifiers"""

    def train(self, X_train, y_train):
        """Train the classifier on the provided dataset

        :param X_train: Training features
        :param y_train: Training labels
        :return: None(the trained model is stored inside the class)
        """
        pass

    def predict(self, X):
        """
        Predict labels for new data

        :param X: Testing features
        :return: Array containing predicted labels
        """
        pass
