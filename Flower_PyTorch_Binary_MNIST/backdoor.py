import flwr as fl
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple

import binary

USE_FEDBN: bool = True

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import random

class PoisonedBinaryClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch with a backdoor attack."""

    def __init__(
        self,
        model: binary.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
        backdoor_trigger_label: int = 1,  # Set the label to trigger the backdoor
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.backdoor_trigger_label = backdoor_trigger_label

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
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        
        # Inject backdoor during training
        self.inject_backdoor()

        # Train the model
        binary.train(self.model, self.trainloader, epochs=5, device=DEVICE)
        
        # Reset backdoor after training
        self.reset_backdoor()

        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def inject_backdoor(self) -> None:
        # Modify a small percentage of training data to have the backdoor trigger label
        for data in self.trainloader:
            inputs, labels = data
            labels = labels.numpy()
            # Inject backdoor into a small percentage of data
            indices_to_poison = random.sample(
                range(len(labels)), int(0.1 * len(labels))
            )
            labels[indices_to_poison] = self.backdoor_trigger_label
            labels = torch.from_numpy(labels)
            break  # Only inject backdoor in the first batch

    def reset_backdoor(self) -> None:
        # Reset backdoor trigger label after training
        self.backdoor_trigger_label = 0  # Set it back to the original trigger label

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = binary.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

def main() -> None:
    """Load data, start PoisonedCifarClient."""

    # Load data
    trainloader, testloader, num_examples = binary.load_data()

    # Load model
    model = binary.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client with backdoor
    client = PoisonedBinaryClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
