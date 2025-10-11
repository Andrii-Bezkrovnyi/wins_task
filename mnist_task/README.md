# Task 1 — MNIST Classification (Random Forest, NN, CNN)

## Description

This task demonstrates image classification on the MNIST dataset using three different
approaches:

Random Forest (sklearn)

Feed-Forward Neural Network (PyTorch)

Convolutional Neural Network (CNN) (PyTorch)

All models are implemented as classes that follow the same OOP interface (
MnistClassifierInterface), and are wrapped by a manager class (MnistClassifier).
This ensures that training and inference APIs remain consistent regardless of the chosen
algorithm.

## **Project Structure**

```text
│
├── mnist_task/
│── demo.ipynb # Jupyter Notebook with model training & demo
│
├── models/
│ │── interface.py # Abstract base class (MnistClassifierInterface)
│ │── rf_classifier.py # Random Forest implementation
│ │── nn_classifier.py # Feed-Forward NN implementation
│ │── cnn_classifier.py # CNN implementation
│ │── mnist_classifier.py # Wrapper class (MnistClassifier)
│
└── utils/
    │── download_data.py # Data downloading and preprocessing helpers
    │── train_eval.py # Training and evaluation functions
│── requirements.txt # Dependencies for Task 1 only
│── dataset/ # MNIST dataset (downloaded automatically)
│── README.md # Documentation

```

## **Setup**

1. Clone or download the project.

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Add kaggle.json to the mnist_task directory of the project.
   You can get kaggle.json from your Kaggle account settings.
    1. Go to the Kaggle website.
    2. Log in to your Kaggle account. If you don't have one, you'll need to create it.
    3. Navigate to your account settings by clicking on your profile icon in the
       top-right.
    4. Scroll down to the “API” section.
    5. Click “Create New API Token” — this will automatically download a file named
       kaggle.json.
    6. This file contains your personal API key required to access Kaggle datasets.
    7. Copy the downloaded kaggle.json file into your project directory `mnist_task/`.
4. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the script `demo.ipynb` in Pycharm or Jupyter Notebook to see the training
and evaluation of all models.

6. Follow the script in `demo.ipynb` for downloading data, train and evaluate each
   model.

### Compare results

```
Accuracy:
Random Forest   : 97.05%
Feed-Forward NN : 95.79%
CNN             : 99.02%
```

### Sample Predictions

```
True	RF	NN	 CNN
 3   	3  	 0	   0
 0   	7	 7	   7
 3   	6	 6	   6
 6      6    6	   6
 3      3    3     3
 8      8    8     8
```

## Summary

- The Random Forest model reached around 97% accuracy, providing a solid benchmark for
  MNIST classification.
- The Feed-Forward Neural Network obtained approximately 96%, demonstrating effective
  learning, though slightly behind due to its simpler structure.
- The Convolutional Neural Network achieved near 99% accuracy, confirming that CNNs are
  highly effective for image recognition tasks.