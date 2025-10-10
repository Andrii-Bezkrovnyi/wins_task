import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .interface import MnistClassifierInterface


class RandomForestClassifierMnist(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST dataset implementing MnistClassifierInterface.

    Attributes:
        model (RandomForestClassifier): scikit-learn RandomForest model instance.
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the Random Forest classifier.

        :param n_estimators: Number of trees in the forest (default=100)
        :param random_state: Random seed for reproducibility (default=42)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Random Forest model.

        :param X_train: Training features (numpy array of shape [n_samples, n_features])
        :param y_train: Training labels (numpy array of shape [n_samples])
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for given input data.

        :param X: Features to predict (numpy array of shape [n_samples, n_features])
        :return: Predicted labels (numpy array)
        """
        return self.model.predict(X)
