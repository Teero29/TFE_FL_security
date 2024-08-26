import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision
from MobileNetV2 import Net, train as mobilenet_train, test as mobilenet_test, load_data

USE_FEDBN: bool = True

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

epochs: int = 25

class MobileNetV2Client(fl.client.NumPyClient):
    """
    Client class for MobileNetV2 model using Flower framework with label flipping data poisoning.
    
    Args:
        model (MobileNetV2.Net): The MobileNetV2 model instance.
        trainloader (torch.utils.data.DataLoader): DataLoader for training dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for test dataset.
        valloader (torch.utils.data.DataLoader): DataLoader for validation dataset.
        num_examples (Dict): Dictionary containing number of examples in each dataset.
    """
    
    def __init__(
        self,
        model: Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        num_examples: Dict[str, int],
    ) -> None:
        super().__init__()
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Get model parameters as a list of NumPy ndarrays.
        
        Args:
            config (Dict[str, str]): Configuration dictionary (not used in this method).
            
        Returns:
            List[np.ndarray]: List of model parameters as NumPy ndarrays.
        """
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of NumPy ndarrays.
        
        Args:
            parameters (List[np.ndarray]): List of model parameters as NumPy ndarrays.
        """
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model with locally poisoned data and return updated parameters.
        
        Args:
            parameters (List[np.ndarray]): List of model parameters as NumPy ndarrays.
            config (Dict[str, str]): Configuration dictionary (not used in this method).
        
        Returns:
            Tuple[List[np.ndarray], int, Dict]: Updated parameters, number of examples in training set, and an empty dictionary.
        """
        poisoned_parameters = self.poison_data(parameters)
        self.set_parameters(poisoned_parameters)
        mobilenet_train(self.model, self.trainloader, self.valloader, epochs=epochs, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(  
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test dataset and return loss, accuracy, and number of test examples.
        
        Args:
            parameters (List[np.ndarray]): List of model parameters as NumPy ndarrays.
            config (Dict[str, str]): Configuration dictionary (not used in this method).
        
        Returns:
            Tuple[float, int, Dict]: Loss, number of examples in test set, and accuracy in a dictionary.
        """
        self.set_parameters(parameters)
        loss, accuracy = mobilenet_test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
        
    def poison_data(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Poison the training data by flipping random labels.
        
        Args:
            parameters (List[np.ndarray]): List of model parameters as NumPy ndarrays (not used in this method).
        
        Returns:
            List[np.ndarray]: List of model parameters as NumPy ndarrays (unchanged in this method).
        """
        for idx, (inputs, labels) in enumerate(self.trainloader):
            flip_rate = 0.1
            num_flips = int(flip_rate * labels.size(0))
            flip_indices = np.random.choice(labels.size(0), num_flips, replace=False)
            labels[flip_indices] = 1 - labels[flip_indices]
            self.trainloader.dataset.targets[idx * self.trainloader.batch_size : (idx + 1) * self.trainloader.batch_size] = labels.numpy()
        return parameters


def main() -> None:
    """
    Main function to initialize and start the MobileNetV2 client.
    """
    trainloader, testloader, valloader, num_examples = load_data()

    model = Net().to(DEVICE).train()
    _ = model(next(iter(trainloader))[0].to(DEVICE))  # Perform a single forward pass

    client = MobileNetV2Client(model, trainloader, testloader, valloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
