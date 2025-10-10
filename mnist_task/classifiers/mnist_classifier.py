from .cnn_classifier import CNNClassifier
from .nn_classifier import FeedForwardNN
from .rf_classifier import RandomForestClassifierMnist


class MnistClassifier:
    """
    A unified wrapper class for selecting and running different MNIST classification classifiers.

    This class allows you to choose between multiple algorithms (Random Forest, Feed-Forward NN, CNN)
    using a single interface for training and prediction.

    Supported algorithms:
        - "rf"  : Random Forest (sklearn)
        - "nn"  : Feed-Forward Neural Network (PyTorch)
        - "cnn" : Convolutional Neural Network (PyTorch)
    """

    def __init__(self, algorithm: str = "rf", **kwargs):
        """
        Initialize the classifier with the chosen algorithm.

        Args:
            algorithm (str): The name of the algorithm to use. One of:
                            - "rf"  : Random Forest
                            - "nn"  : Feed-Forward Neural Network
                            - "cnn" : Convolutional Neural Network
            **kwargs: Additional parameters passed to the model's constructor
                      (e.g., n_estimators for RF, epochs or batch_size for NN/CNN).

        Raises:
            ValueError: If an unknown algorithm name is provided.
        """
        match algorithm:
            case "rf":
                self.model = RandomForestClassifierMnist(**kwargs)
            case "nn":
                self.model = FeedForwardNN(**kwargs)
            case "cnn":
                self.model = CNNClassifier(**kwargs)
            case _:
                raise ValueError(
                    "There is no algorithm. Choose from: 'rf', 'nn', 'cnn'."
                )

    def train(self, X_train, y_train):
        """
        Train the selected MNIST model.

        Args:
            X_train: Training features. For:
                     - Random Forest: numpy array of shape (n_samples, 784).
                     - NN / CNN: torch.Tensor of shape (n_samples, 1, 28, 28).
            y_train: Training labels. numpy array or torch.Tensor of shape (n_samples,).

        Returns:
            None
        """
        return self.model.train(X_train, y_train)

    def predict(self, X):
        """
        Predict labels using the trained model.

        Args:
            X: Input data for prediction. Format depends on the chosen model:
               - Random Forest: numpy array of shape (n_samples, 784).
               - NN / CNN: torch.Tensor of shape (n_samples, 1, 28, 28).

        Returns:
            Predicted labels (numpy array or torch.Tensor).
        """
        return self.model.predict(X)
