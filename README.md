# Federated Learning Pipeline with Flower and PyTorch
This repository implements a Federated Learning (FL) pipeline using Flower (FLWR), PyTorch, and Hydra for configuration management. The system allows for distributed training of machine learning models across multiple clients, using MNIST as the example dataset, with decentralized data storage and local training. This project is based on the tuorial made by FLOWER to simulate a federated learning pipeline (https://github.com/adap/flower/tree/main/examples/flower-simulation-step-by-step-pytorch)

# Features
  Federated Learning: Uses Flower for managing federated learning simulations and model aggregation.
  Custom Models: Users can define and train their own model architectures.
  Client Simulation: Simulates multiple clients, each with its own subset of the dataset for local training.
  Dataset Handling: Efficient loading and partitioning of the MNIST dataset into training and validation sets.
  Configurable Parameters: Hyperparameters (learning rate, momentum, local epochs, etc.) can be customized via Hydra configurations.
  GPU Support: The code automatically detects and uses available GPUs for training if present.

# File Structure
- fit_strategy.py: Contains functions for generating the configuration for training (fit) and evaluation during federated learning rounds.
- model.py: Defines the neural network architecture (Net) and the functions for training and testing the model on the MNIST dataset.
- server.py: Sets up the server-side strategy for federated learning, including model aggregation and starting the simulation.
- client.py: Defines the FlowerClient class, which handles local training, updating model parameters, and evaluating the model on a client’ dataset.
- dataset.py: Contains functions to load and split the MNIST dataset into partitions for clients, with IID data splitting strategy.

# Dataset
  This pipeline uses the MNIST dataset as an example. The dataset is downloaded automatically using PyTorch's torchvision.datasets.MNIST API. The data is split into num_partitions for client training and further split into training and validation sets for each client.
  
  prepare_dataset: The prepare_dataset function partitions the MNIST dataset into client-specific training and validation sets, with the option to specify the number of partitions (clients) and the batch size.

 # How It Works
  Federated Learning Process:
  Server Initialization:
  - The server orchestrates the federated learning rounds, where clients receive a global model, perform local training on their dataset partitions, and send back the updated model parameters. The server aggregates these updates to refine the global model.
  
  Client Training:
  - Each client uses its own subset of the dataset for local training. The FlowerClient class is responsible for handling:
  
  - Setting parameters received from the server.
  - Training the model locally using the MNIST data partition.
  - Sending the updated model parameters back to the server after local training.
  Model Aggregation:
  - After each round, the server aggregates model updates from clients to create an improved global model.
