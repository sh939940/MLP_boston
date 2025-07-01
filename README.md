# 🏠 MLP Boston Housing Price Prediction

This project demonstrates a complete machine learning workflow using a **Multi-Layer Perceptron (MLP)** model to predict housing prices on the classic Boston Housing dataset.

## 📁 Project Structure

- **MLP_boston.ipynb**: Trains an MLP model using TensorFlow/Keras on the Boston Housing dataset.
- **load_deploy.ipynb**: Loads the saved `.h5` model and demonstrates how to make predictions on new data.

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn

## 📊 Dataset

The **Boston Housing** dataset is a classic dataset from the UCI Machine Learning Repository. It includes features like crime rate, number of rooms, accessibility to highways, etc.

## 🧠 Model Summary

- **Model Type**: Multi-Layer Perceptron (MLP)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Metrics**: MSE

## ▶️ How to Use

1. Run `MLP_boston.ipynb` to train and save the model.
2. Run `load_deploy.ipynb` to load the saved model and make predictions.

## 📦 Output

The model predicts the median house price (in $1000s) based on 13 input features. Before prediction, the input data is standard scaled to match the format used during model training. 
The final output is a single numerical value representing the estimated house price for the given input.

