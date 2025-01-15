#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

Do a multi-classification task to determine the position of a specific character in a string using RNN and cross entropy.

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding Layer
        self.rnn = nn.LSTM(input_size=vector_dim, hidden_size=vector_dim, batch_first=True)  # LSTM Layer
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # Linear layer for classification
        self.loss = nn.CrossEntropyLoss()  # Cross-entropy loss function

    # When the true label is entered, the loss value is returned; if there is no true label, the predicted value is returned
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)                         # (batch_size, sen_len, vector_dim)
        x = x[:, -1, :]                            # Take the output of the last time step (batch_size, vector_dim)                        #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = self.classify(x)                       # (batch_size, vector_dim) -> (batch_size, sentence_length+1)
        if y is not None:
            return self.loss(x, y)                 #compute loss
        else:
            return torch.argmax(x, dim=1)          # Return predicted class indices

# The character set randomly picked some characters, which can actually be expanded
# Generate a label for each character
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "youmeheshedefghijklmnopqrstuvwxyz"  #Character Set
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   # Each word corresponds to a serial number
    vocab['unk'] = len(vocab) #26
    return vocab

# Randomly generate a sample
# Select sentence_length words from all words
# Otherwise it is a negative sample
def build_sample(vocab, sentence_length, target_char="你"):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if target_char in x:
        y = x.index(target_char)  # Position of the target character
    else:
        y = sentence_length  # Class representing "not found"
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

# Create a data set
# Enter the number of samples required. Generate as many as needed
def build_dataset(sample_length, vocab, sentence_length, target_char="你"):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_char)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# Build model
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# Test code
# Used to test the accuracy of each round of model
def evaluate(model, vocab, sentence_length, target_char="you"):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length, target_char)   # Create 200 samples for testing
    correct = 0
    with torch.no_grad():
        y_pred = model(x)      # Model prediction
        correct = (y_pred == y).sum().item()  # Calculate the number of correct predictions
    accuracy = correct / len(y)  # Calculate accuracy
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy


def main():
    # Configuration parameters
    epoch_num = 10        # Number of training rounds
    batch_size = 20       # Number of training samples each time
    train_sample = 500    # Total number of samples trained in each round of training
    char_dim = 20         # The dimension of each word
    sentence_length = 6   # Sample text length
    learning_rate = 0.005 # Learning rate
    target_char = "you"    # Character to detect
    # Create a word list
    vocab = build_vocab()
    # Build the model
    model = build_model(vocab, char_dim, sentence_length)
    # Select optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length, target_char) # Construct a set of training samples
            optim.zero_grad()    # Gradient zeroing
            loss = model(x, y)   # Calculate loss
            loss.backward()      # Calculate gradients
            optim.step()         # Update weights
            watch_loss.append(loss.item())
        print("=========\nAverage loss in round %d:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, target_char)   # Test the model results for this round
        log.append([acc, np.mean(watch_loss)])
    # Draw the accuracy and loss curves
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")  # Draw acc curve
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss") # Draw the loss curve
    plt.legend()
    plt.show()
    # Save model
    torch.save(model.state_dict(), "model.pth")
    # Save the word list
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#Use the trained model to make predictions
def predict(model_path, vocab_path, input_strings, target_char="you"):
    char_dim = 20  # The dimension of each word
    sentence_length = 6  #  Sample text length
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) # Load character table
    model = build_model(vocab, char_dim, sentence_length)     # Build model
    model.load_state_dict(torch.load(model_path))             # Load trained weights
    model.eval()

    x = []
    for input_string in input_strings:
        # Pad or truncate each input string
        input_string = input_string[:sentence_length]  # Truncate if too long
        padded_string = input_string + " " * (sentence_length - len(input_string))
        x.append([vocab.get(char, vocab['unk']) for char in padded_string])  # Serialize input

    x = torch.LongTensor(x)

    with torch.no_grad():  # Do not calculate gradients
        predictions = model(x)  # Get predictions
    for i, input_string in enumerate(input_strings):
        predicted_class = predictions[i].item()
        if predicted_class == sentence_length:
            print(f"Input: {input_string}, Prediction: '{target_char}' not found")
        else:
            print(f"Input: {input_string}, Prediction: '{target_char}' found at position {predicted_class}")

if __name__ == "__main__":
    main()
    test_strings = ["abcdyou", "meabcdef", "xyzuvw", "defghme"]
    predict("model.pth", "vocab.json", test_strings)
