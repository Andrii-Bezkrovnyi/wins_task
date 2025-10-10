# Task 1 — MNIST Classification (Random Forest, NN, CNN)

## Description

This task demonstrates image classification on the MNIST dataset using three different approaches:

Random Forest (sklearn)

Feed-Forward Neural Network (PyTorch)

Convolutional Neural Network (CNN) (PyTorch)

All models are implemented as classes that follow the same OOP interface (MnistClassifierInterface), and are wrapped by a manager class (MnistClassifier).
This ensures that training and inference APIs remain consistent regardless of the chosen algorithm.

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
3. Install the required libraries:
   ```bash
   pip install -rm requirements.txt
   ```
4. Run the script `demo.ipynb` in Pycharm or Jupyter Notebook to see the training 
and evaluation of all models. 

5. Follow the on-screen menu and enjoy!


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
3	3  	 0	   0
0	7	 7	   7
3	6	 6	   6
6       6        6	   6
3       3        3         3
8       8        8         8
```

## Summary

- The Random Forest model reached around 97% accuracy, providing a solid benchmark for MNIST classification.
- The Feed-Forward Neural Network obtained approximately 96%, demonstrating effective learning, though slightly behind due to its simpler structure.
- The Convolutional Neural Network achieved near 99% accuracy, confirming that CNNs are highly effective for image recognition tasks.