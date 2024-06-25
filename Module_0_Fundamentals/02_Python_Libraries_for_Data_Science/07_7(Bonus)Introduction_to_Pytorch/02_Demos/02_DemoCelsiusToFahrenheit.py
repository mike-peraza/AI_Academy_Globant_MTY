############################################################################################
# Example to convert Celsius (C) to Fahrenheit (F)
# F = C * 1.8 + 32

# Regular programing
# def function(C):
#     F = C * 1.8 + 32
#     return F
############################################################################################

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define a simple neural network model
class CelsiusToFahrenheitModel(nn.Module):
    def __init__(self):
        super(CelsiusToFahrenheitModel, self).__init__()
        # Define a linear layer with one input and one output
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Function to train the model
def train_model(model, inputs, targets, learning_rate=0.001, epochs=1000):
    # Define a loss function (Mean Squared Error)
    criterion = nn.MSELoss()
    # Define an optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # List to store loss values for each epoch
    loss_values = []
    
    # Training loop
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        # Forward pass: Compute predicted y by passing x to the model
        predictions = model(inputs)
        # Compute the loss
        loss = criterion(predictions, targets)
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the loss value
        loss_values.append(loss.item())
        
        # Print the loss every 100 epochs
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            # Print out the current parameters of the model
            for name, param in model.named_parameters():
                print(f"{name}: {param.data}")

        # Check for NaN in loss and exit if encountered
        if torch.isnan(loss):
            print(f"NaN encountered in epoch {epoch+1}. Stopping training.")
            break
    
    print("Training complete!")

    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# Function to convert Celsius to Fahrenheit using the trained model
def celsius_to_fahrenheit(celsius_values, model):
    # Convert the input to a PyTorch tensor and reshape it
    celsius_tensor = torch.tensor(celsius_values, dtype=torch.float32).reshape(-1, 1)
    # Predict Fahrenheit values
    with torch.no_grad():  # No need to compute gradients for inference
        fahrenheit_tensor = model(celsius_tensor)
    return fahrenheit_tensor

# Example usage
# Training data: Celsius to Fahrenheit pairs
celsius_samples = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=np.float32)
fahrenheit_samples = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=np.float32)

# Convert the training data to PyTorch tensors
# Reshape to (n_samples, 1)
inputs = torch.tensor(celsius_samples).reshape(-1, 1)
targets = torch.tensor(fahrenheit_samples).reshape(-1, 1)

# Initialize the model
model = CelsiusToFahrenheitModel()

# Train the model
print("Training Model...")
trained_model = train_model(model, inputs, targets, learning_rate=0.001)

# Convert new Celsius values using the trained model
new_celsius_values = [0, 20, 100, -40]  # List of new Celsius values to convert
fahrenheit_values = celsius_to_fahrenheit(new_celsius_values, trained_model)

print("Celsius values:", new_celsius_values)
# Converting tensor to NumPy array for easy reading
print("Converted Fahrenheit values:", fahrenheit_values.numpy().flatten())
