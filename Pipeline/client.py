import flwr as fl
import torch
from collections import OrderedDict
from model import Net, train, test
from flwr.common import NDArrays, Scalar
from typing import Dict, Tuple

# FlowerClient class inherits from Flower's NumPyClient
class FlowerClient(fl.client.NumPyClient):
    
    # Initialize the client with trainloader, validation loader, and number of classes
    def __init__(self, trainloader, validation_loader, num_classes) -> None:
        super().__init__()  # Initialize the parent class
        self.trainloader = trainloader  # Store the training data loader
        self.validation_loader = validation_loader  # Store the validation data loader
        self.model = Net(num_classes)  # Initialize the neural network model with num_classes output
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set the device (GPU if available)

    # Update local model parameters with the ones received from the server
    def set_parameters(self, parameters):
        # Zip together model state_dict keys with the received parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # Convert parameters into a state dictionary format
        state_dict = OrderedDict({k: torch.Tensor(v) for k,v in params_dict})
        # Load the parameters into the model
        self.model.load_state_dict(state_dict, strict=True)

    # Get the local model's parameters and return as a list of numpy arrays
    def get_parameters(self, config: Dict[str, Scalar]):
        # Convert the model's parameters into a list of numpy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Perform local training on the client's data
    def fit(self, parameters, config):
        print(f"config dict contains : {config}")
        # Update the model with server-sent parameters
        self.set_parameters(parameters)

        # Extract learning rate, momentum, and local epochs from the config
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_num_epochs']
        print(epochs)
        # Initialize the optimizer with the specified learning rate and momentum
        opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        # Train the model locally using the provided train function
        train(self.model, self.trainloader, opt, epochs, self.device)

        # Return updated parameters, size of the local dataset, and an empty dict for additional info
        return self.get_parameters({}), len(self.trainloader), {}
    
    # Evaluate the local model on the client's validation dataset
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Update the model with server-sent parameters before evaluation
        self.set_parameters(parameters)

        # Evaluate the model using the validation data and return loss and accuracy
        loss, accuracy = test(self.model, self.validation_loader, self.device)

        # Return the loss, validation dataset size, and a dict containing the accuracy
        return float(loss), len(self.validation_loader), {"accuracy": accuracy}


# Function to generate a client function for the federated learning simulation
def generate_client_fn(trainloader, validation_loader, num_classes):
    """Return a function that can be used by the VirtualClientEngine to spawn FlowerClients.

    This client function will be called internally during the simulation when a specific
    client (by its client ID) is requested to participate in federated learning.
    """

    def client_fn(cid: str):
        # This function is called by the federated learning engine and uses the client ID (cid)
        # to return a FlowerClient with the cid-th train and validation loaders
        return FlowerClient(
            trainloader=trainloader[int(cid)],  # Assign the cid-th training data
            validation_loader=validation_loader[int(cid)],  # Assign the cid-th validation data
            num_classes=num_classes,  # Pass the number of output classes to the client
        )

    # Return the client function that will be used to spawn clients
    return client_fn



    
      
