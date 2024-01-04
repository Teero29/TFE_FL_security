
import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple
import torch.nn.functional as F
import torchvision.transforms as transforms
import flwr as fl
import numpy as np
import torch
import torchvision
import random
import binary

USE_FEDBN: bool = True

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

# Flower Client
class BinaryAttackClient(fl.client.NumPyClient):

    def __init__(
        self,
        model: binary.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
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
        # Set model parameters from a list of NumPy ndarrays
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
        # Set model parameters with poisoned data, train model, return updated model parameters
        self.set_parameters(parameters)

        # Apply label flipping attack
        flip_percentage = 0.1 
        self.apply_label_flipping_attack(flip_percentage)

        binary.train(self.model, self.trainloader, epochs=5, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}
    
    def apply_label_flipping_attack(self, flip_percentage: float) -> None:
        # Randomly flip labels in the training dataset
        for i, (data, labels) in enumerate(self.trainloader):
            num_flips = int(flip_percentage * len(labels))
            flip_indices = random.sample(range(len(labels)), num_flips)
            for flip_index in flip_indices:
                labels[flip_index] = 1 - labels[flip_index]  # Flip the label

            # Update the labels in the dataset
            self.trainloader.dataset.targets[i * self.trainloader.batch_size : (i + 1) * self.trainloader.batch_size] = labels

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = binary.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

def main() -> None:
    """Load data, start CifarClient."""

    # Load data
    trainloader, testloader, num_examples = binary.load_data()
    # Load model
    model = binary.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = BinaryAttackClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
