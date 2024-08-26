import os
import sys
from typing import Dict, List, Tuple

import torch
import numpy as np
import flwr as fl
from collections import OrderedDict
import MobileNetV2


USE_FEDBN: bool = True

# Determine device (GPU/CPU)
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs: int = 25

class MobileNetV2Client(fl.client.NumPyClient):
    """
    Client implementation for MobileNetV2 model for Federated Learning.
    
    Attributes:
        model (MobileNetV2.Net): The MobileNetV2 model instance.
        trainloader (torch.utils.data.DataLoader): DataLoader for training dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for test dataset.
        valloader (torch.utils.data.DataLoader): DataLoader for validation dataset.
        num_examples (Dict[str, int]): Number of examples in each dataset partition.
    """
    
    def __init__(
        self,
        model: MobileNetV2.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        num_examples: Dict[str, int],
    ) -> None:
        """
        Initialize the client with model and data loaders.
        
        Args:
            model (MobileNetV2.Net): The MobileNetV2 model instance.
            trainloader (torch.utils.data.DataLoader): DataLoader for training dataset.
            testloader (torch.utils.data.DataLoader): DataLoader for test dataset.
            valloader (torch.utils.data.DataLoader): DataLoader for validation dataset.
            num_examples (Dict[str, int]): Number of examples in each dataset partition.
        """
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Get model parameters as a list of NumPy ndarrays.
        
        Args:
            config (Dict[str, str]): Configuration parameters (not used currently).
        
        Returns:
            List[np.ndarray]: List of model parameters as NumPy ndarrays.
        """
        self.model.train()
        if USE_FEDBN:
            # Exclude parameters of BatchNorm layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
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
        else:
            keys = self.model.state_dict().keys()
        
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=not USE_FEDBN)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data and return updated model parameters.
        
        Args:
            parameters (List[np.ndarray]): List of model parameters as NumPy ndarrays.
            config (Dict[str, str]): Configuration parameters (not used currently).
        
        Returns:
            Tuple[List[np.ndarray], int, Dict]: Updated model parameters, number of training examples, and empty dictionary.
        """
        self.set_parameters(parameters)
        MobileNetV2.train(self.model, self.trainloader, self.valloader, epochs=epochs, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data and return loss, accuracy, and number of test examples.
        
        Args:
            parameters (List[np.ndarray]): List of model parameters as NumPy ndarrays.
            config (Dict[str, str]): Configuration parameters (not used currently).
        
        Returns:
            Tuple[float, int, Dict]: Loss, number of test examples, and accuracy in a dictionary.
        """
        self.set_parameters(parameters)
        loss, accuracy = MobileNetV2.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    """
    Main function to start the federated learning client.
    """
    # Load data
    trainloader, testloader, valloader, num_examples = MobileNetV2.load_data()

    # Load model
    model = MobileNetV2.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = MobileNetV2Client(model, trainloader, testloader, valloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
