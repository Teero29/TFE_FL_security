import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision
import MobileNetV2


USE_FEDBN: bool = True

# Determine device (CPU/GPU)
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs: int = 25


class MobileNetV2Client(fl.client.NumPyClient):
    """
    Client class for federated learning using MobileNetV2 model.
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
        Initialize the MobileNetV2 client.

        Parameters:
        ----------
        model : MobileNetV2.Net
            The MobileNetV2 model instance.
        trainloader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        testloader : torch.utils.data.DataLoader
            DataLoader for the test dataset.
        valloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        num_examples : Dict[str, int]
            Dictionary containing the number of examples for each dataset.

        Returns:
        -------
        None
        """
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Get the model parameters as a list of NumPy ndarrays.

        Parameters:
        ----------
        config : Dict[str, str]
            Configuration dictionary (not used in this method).

        Returns:
        -------
        List[np.ndarray]
            List of NumPy ndarrays representing the model parameters.
        """
        self.model.train()
        if USE_FEDBN:
            # Exclude parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the model parameters from a list of NumPy ndarrays.

        Parameters:
        ----------
        parameters : List[np.ndarray]
            List of NumPy ndarrays representing the model parameters.

        Returns:
        -------
        None
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
        Train the model on the local dataset.

        Parameters:
        ----------
        parameters : List[np.ndarray]
            List of NumPy ndarrays representing the model parameters.
        config : Dict[str, str]
            Configuration dictionary (not used in this method).

        Returns:
        -------
        Tuple[List[np.ndarray], int, Dict]
            Tuple containing updated model parameters, number of examples used,
            and an empty dictionary (not used).
        """
        self.add_noise_to_trainset()
        MobileNetV2.train(self.model, self.trainloader, self.valloader, epochs=epochs, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on the local test dataset.

        Parameters:
        ----------
        parameters : List[np.ndarray]
            List of NumPy ndarrays representing the model parameters.
        config : Dict[str, str]
            Configuration dictionary (not used in this method).

        Returns:
        -------
        Tuple[float, int, Dict]
            Tuple containing loss, number of test examples used, and a dictionary
            with the accuracy value.
        """
        self.set_parameters(parameters)
        loss, accuracy = MobileNetV2.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def add_noise_to_trainset(self) -> None:
        """
        Add noise to the training set inputs.
        
        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        for data in self.trainloader:
            inputs, labels = data
            inputs += torch.randn_like(inputs) * 0.9  # Add noise with std 0.9 to inputs

def main() -> None:
    """
    Main function to initialize and start the MobileNetV2 client for federated learning.
    
    Parameters:
    ----------
    None

    Returns:
    -------
    None
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
