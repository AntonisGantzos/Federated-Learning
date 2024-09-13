from omegaconf import DictConfig
from collections import OrderedDict
from model import Net, train, test
import torch

# Generate the configuration function for training (fitting)
def get_on_fit_config(config: DictConfig):
    """
    This function returns a configuration function for the local training (fit) process,
    where hyperparameters like learning rate, momentum, and number of epochs are specified.
    
    Args:
        config: A DictConfig object containing the training hyperparameters.
        
    Returns:
        fit_config_fn: A function that provides the hyperparameters to the client during training.
    """
    # Function to return fit configuration for each training round
    def fit_config_fn(server_round: int):
        # Extract hyperparameters from the passed config and return them in a dictionary
        return {'lr': config.lr, 'momentum': config.momentum, 'local_num_epochs': config.local_epochs}
    
    # Return the configuration function
    return fit_config_fn

# Generate an evaluation function for model testing
def get_evaluate_fn(num_classes: int, testloader):
    """
    This function returns an evaluation function that tests the model on a given test dataset,
    and calculates the loss and accuracy.
    
    Args:
        num_classes: The number of output classes for the neural network.
        testloader: DataLoader containing the test dataset.
        
    Returns:
        evaluate_fn: A function to evaluate the model using the test set.
    """
    # Function to evaluate the model during a specific federated learning round
    def evaluate_fn(server_round: int, parameters, config):
        # Initialize a new neural network model with the specified number of classes
        model = Net(num_classes)

        # Set the device (GPU if available, otherwise CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the parameters sent by the server into the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k,v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Use the test function to evaluate the model and return the loss and accuracy
        loss, accuracy = test(model, testloader, device)
        
        # Return the loss and a dictionary containing accuracy information
        return loss, {'accuracy': accuracy}
    
    # Return the evaluation function
    return evaluate_fn
