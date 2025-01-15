# Multi-Class Classification with RNN and Cross-Entropy

## Project Overview
This project demonstrates how to train a multi-class classification model using a Recurrent Neural Network (RNN) and cross-entropy loss to determine the position of a specific character in a string. The model is trained to predict the index of a target character within a given string or indicate that the character is not present.

## Key Components
- **TorchModel**: A PyTorch model that includes an embedding layer, an LSTM layer, and a linear classification layer.
- **build_vocab**: A function to create a vocabulary mapping characters to indices.
- **build_sample**: A function to generate a single training sample.
- **build_dataset**: A function to generate a dataset of multiple training samples.
- **evaluate**: A function to evaluate the model's accuracy on a test set.
- **main**: The main function to train the model and save the trained model and vocabulary.
- **predict**: A function to use the trained model to make predictions on new input strings.

## Setup and Installation
1. **Install PyTorch**: Ensure you have PyTorch installed. You can install it using pip:
   ```sh
   pip install torch
