###########################################################################
#In federated learning, there are several ways to split a dataset across clients, primarily depending on the problem and type of data distribution:
#
#IID (Independent and Identically Distributed): The data is randomly shuffled and equally distributed among clients.
#Non-IID (Non-Independent and Non-Identically Distributed): Data is split based on different distributions (e.g., each client may only have access to a subset of data categories or features).
#Horizontal Federated Learning: Clients share the same feature space but have different samples.
#Vertical Federated Learning: Clients share the same samples but have different feature spaces.
#These methods adapt based on specific learning objectives.
#
#1. IID (Independent and Identically Distributed)
#Example: Suppose you have a dataset with 10,000 images of handwritten digits from 0 to 9 (e.g., MNIST). In IID, the data is shuffled randomly, and each client gets a balanced portion, like 1,000 images with all digits roughly equally represented.
#2. Non-IID
#Example: In the same MNIST dataset, one client might receive only images of digits 0-2, another client gets images of digits 3-5, and so on. Each client has data biased towards a specific category.
#3. Horizontal Federated Learning
#Example: Multiple hospitals share patient data, but each hospital has a different set of patients (rows) with the same features (columns, like age, blood pressure). The dataset is split horizontally, based on patient records.
#4. Vertical Federated Learning
#Example: Two organizations (e.g., a hospital and an insurance company) share data on the same patients, but each holds different features (e.g., the hospital has medical history, while the insurance company has financial information). The split is based on features, with the same entities (patients) appearing on both sides.
#These methods allow federated learning to handle different types of data splits based on the needs of the clients.
############################################################################

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
import torch


#get MNIST dataset from pytorch library and split it to train and test datasets
def getMNIST(data_path : str = "./data"):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(data_path, train = True, download = True, transform = tr)
    testset = MNIST(data_path, train = False, download = True, transform = tr)
    return trainset, testset




#split each dataset into dataloaders 
#we will adopt the IID strategy for the split of our data
def prepare_dataset(
        num_partitions : int, 
        batch_size:int,
        val_ratio : float = 0.1
):
    trainset, testset = getMNIST()

    #split trainset into num_partitions trainsets
    num_images = len(trainset) // num_partitions

    #This creates a list called partition_len, which contains num_partitions entries, each with the value num_images.
    #This list represents the size of each partition.
    partition_len = [num_images] * num_partitions

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(42))


    #create dataloaders with train + val support
    trainloader = []
    validation_loader = []
    for set in trainsets:
        #get the length of each batch
        num_total= len(set)
        #get the validation sample
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(set, [num_train, num_val], torch.Generator().manual_seed(42))

        trainloader.append(DataLoader(for_train, batch_size=batch_size, shuffle=True))
        validation_loader.append(DataLoader(for_val, batch_size=batch_size, shuffle=False))

    testloader = DataLoader(testset, batch_size=128)

    return trainloader, validation_loader, testloader