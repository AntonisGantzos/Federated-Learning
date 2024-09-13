import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a neural network class 'Net' inheriting from nn.Module
class Net(nn.Module):
    # Initialize the layers of the network
    def __init__(self, num_classes:int) -> None:
        super(Net, self).__init__()
        # 1 input channel (grayscale), 6 output channels, 5x5 convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Max pooling layer with a 2x2 window
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolution: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected (linear) layer: input size = 16 * 4 * 4 (after convolution and pooling), output size = 120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # Second fully connected layer: input size = 120, output size = 84
        self.fc2 = nn.Linear(120, 84)
        # Final fully connected layer: input size = 84, output size = number of classes (num_classes)
        self.fc3 = nn.Linear(84, num_classes)

    # Define the forward pass of the network
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Apply first convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layers (view reshapes the tensor)
        x = x.view(-1, 16 * 4 * 4)
        # Pass through the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Pass through the second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Output layer without activation (as it's typically combined with a loss function like softmax)
        x = self.fc3(x)
        return x
    
def train(net, trainloader, optimizer, epochs, device: str):
    """
    Train the network on the training set.

    Args:
        net: The neural network model.
        trainloader: DataLoader containing the training data.
        optimizer: Optimization algorithm.
        epochs: Number of training epochs.
        device: Device to use for training ('cpu' or 'cuda').

    This is a fairly simple training loop for PyTorch.
    """
    # Define the loss function (Cross-Entropy Loss for classification)
    criterion = torch.nn.CrossEntropyLoss()

    # Set the network to training mode
    net.train()

    # Move the model to the specified device (CPU or GPU)
    net.to(device)

    # Loop over the number of epochs
    for _ in range(int(epochs)):
        # Iterate through the training data batches
        for images, labels in trainloader:
            # Move the images and labels to the specified device
            images, labels = images.to(device), labels.to(device)

            # Reset the gradients of the optimizer
            optimizer.zero_grad()

            # Forward pass: Compute predicted labels by passing images through the network
            loss = criterion(net(images), labels)

            # Backward pass: Compute gradients with respect to the loss
            loss.backward()

            # Update model parameters using the optimizer
            optimizer.step()



def test(net, testloader, device: str):
    """
    Validate the network on the entire test set and report loss and accuracy.

    Args:
        net: The neural network model.
        testloader: DataLoader containing the test data.
        device: Device to use for testing ('cpu' or 'cuda').

    Returns:
        loss: Total loss over the test set.
        accuracy: Accuracy of the model on the test set.
    """
    # Define the loss function (Cross-Entropy Loss for classification)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize variables for tracking the total correct predictions and loss
    correct, loss = 0, 0.0

    # Set the network to evaluation mode (no dropout or batch norm updates)
    net.eval()

    # Move the model to the specified device (CPU or GPU)
    net.to(device)

    # Disable gradient computation (speeds up testing)
    with torch.no_grad():
        # Iterate through the test data batches
        for data in testloader:
            # Move the images and labels to the specified device
            images, labels = data[0].to(device), data[1].to(device)

            # Forward pass: Get the network's output predictions for the images
            outputs = net(images)

            # Accumulate the total loss for all test batches
            loss += criterion(outputs, labels).item()

            # Get the predicted labels by taking the maximum value from the output
            _, predicted = torch.max(outputs.data, 1)

            # Accumulate the number of correct predictions
            correct += (predicted == labels).sum().item()

    # Calculate accuracy as the ratio of correct predictions to total test examples
    accuracy = correct / len(testloader.dataset)

    # Return the total loss and accuracy over the test set
    return loss, accuracy
