"""Flower client example using PyTorch for CIFAR-10 image classification."""

import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import flwr as fl
import numpy as np
import torch
import torchvision

import binary

USE_FEDBN: bool = True

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

# Flower Client
class BinaryAttackClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

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
        # Add noise to the training set, train model, return updated model parameters
        self.add_noise_to_trainset()
        binary.train(self.model, self.trainloader, epochs=5, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = binary.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def add_noise_to_trainset(self) -> None:
        # Add noise to the training set
        for data in self.trainloader:
            inputs, labels = data
            inputs += torch.randn_like(inputs) * 0.1  # Add Gaussian noise with std 0.5 to inputs
def brighten_images(images, factor=1.5):
    # Brighten the images
    brightened_images = torch.clamp(images * factor, 0, 1)
    return brightened_images

def to_grayscale(images):
    # Convert images to grayscale
    grayscale_images = torch.mean(images, dim=1, keepdim=True)
    return grayscale_images

def plot_images(data_loader, num_images=5, brighten_factor=1.5):
    # Get a batch of images from the data loader
    data_iter = iter(data_loader)
    images, _ = next(data_iter)

    # Convert images to grayscale
    grayscale_images = to_grayscale(images)

    # Brighten grayscale images
    brightened_images = brighten_images(grayscale_images, factor=brighten_factor)

    # Plot the images
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    for i in range(num_images):
        # Plot original images (in grayscale)
        axes[0, i].imshow(np.squeeze(grayscale_images[i].numpy(), axis=0), cmap='gray')
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        # Plot brightened images (in grayscale)
        axes[1, i].imshow(np.squeeze(brightened_images[i].numpy(), axis=0), cmap='gray')
        axes[1, i].axis("off")
        axes[1, i].set_title("Brightened")

    plt.show()

def main() -> None:
    """Load data, start Client."""

    # Load data
    trainloader, testloader, num_examples = binary.load_data()

    # Load model
    model = binary.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = BinaryAttackClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

def showdata() -> None:

    # Load data
    trainloader, testloader, num_examples = binary.load_data()
    plot_images(trainloader)

if __name__ == "__main__":
    showdata()
    #main()
