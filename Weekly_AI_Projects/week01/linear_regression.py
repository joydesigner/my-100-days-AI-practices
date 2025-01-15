import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


# Function to generate data
def generate_data():
    np.random.seed(42)  # For reproducibility
    x = np.random.rand(100, 1)  # Generate 100 random numbers
    y = 2 * x + 1 + 0.1 * np.random.randn(100, 1)  # Linear relationship y = 2x + 1 + noise
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


# Function to visualize data
def visualize_data(x, y, predictions=None):
    plt.scatter(x, y, label='Original data')
    if predictions is not None:
        plt.plot(x, predictions, color='red', label='Fitted line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Input feature dimension is 1, output dimension is 1

    def forward(self, x):
        return self.linear(x)


# Function to train the model
def train_model(model, x, y, num_epochs=1000, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass
        predictions = model(x)

        # Compute loss
        loss = criterion(predictions, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        # Print loss
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


# Function to evaluate the model
def evaluate_model(model, x, y):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(x)
        test_loss = nn.MSELoss()(predictions, y)
        print(f'Test Loss: {test_loss.item():.4f}')
    return predictions


# Main function
def main():
    # Generate data
    x, y = generate_data()

    # Visualize the generated data
    visualize_data(x, y)

    # Initialize the model
    model = LinearRegressionModel()

    # Train the model
    trained_model = train_model(model, x, y)

    # Evaluate the model
    predictions = evaluate_model(trained_model, x, y)

    # Visualize the training results
    visualize_data(x, y, predictions.detach().numpy())


if __name__ == "__main__":
    main()
