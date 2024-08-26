import os
import sys
from typing import Dict, List, Tuple
import random
from collections import OrderedDict

import torch
import numpy as np
import flwr as fl
from MobileNetV2 import Net, load_data, train, test

# Global flag for Federated Batch Normalization (FedBN)
USE_FEDBN: bool = True

# Device selection
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of epochs for training
epochs: int = 25


class MobileNetV2Client(fl.client.NumPyClient):
    """
    Flower client with MobileNetV2 model for Federated Learning with backdoor.

    Attributes:
        model (MobileNetV2.Net): The MobileNetV2 model instance.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.
        valloader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_examples (Dict[str, int]): Dictionary containing number of examples for each dataset.
        backdoor_trigger_label (int): Label used to trigger the backdoor in training.
        brighten_factor (float): Factor to brighten images injected with backdoor trigger label.
    """

    def __init__(
        self,
        model: Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        num_examples: Dict[str, int],
        backdoor_trigger_label: int = 1,
        brighten_factor: float = 0.5,
    ) -> None:
        """
        Initialize MobileNetV2Client with model and data loaders.

        Args:
            model (MobileNetV2.Net): The MobileNetV2 model instance.
            trainloader (torch.utils.data.DataLoader): DataLoader for training data.
            testloader (torch.utils.data.DataLoader): DataLoader for test data.
            valloader (torch.utils.data.DataLoader): DataLoader for validation data.
            num_examples (Dict[str, int]): Dictionary containing number of examples for each dataset.
            backdoor_trigger_label (int, optional): Label used to trigger the backdoor in training. Defaults to 1.
            brighten_factor (float, optional): Factor to brighten images injected with backdoor trigger label. Defaults to 0.5.
        """
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.backdoor_trigger_label = backdoor_trigger_label
        self.brighten_factor = brighten_factor

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Get model parameters as a list of NumPy ndarrays.

        Args:
            config (Dict[str, str]): Configuration dictionary (not used).

        Returns:
            List[np.ndarray]: List of NumPy ndarrays representing model parameters.
        """
        self.model.train()
        if USE_FEDBN:
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
            parameters (List[np.ndarray]): List of NumPy ndarrays representing model parameters.
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
        Train the model with given parameters and return updated parameters after training.

        Args:
            parameters (List[np.ndarray]): List of NumPy ndarrays representing model parameters.
            config (Dict[str, str]): Configuration dictionary (not used).

        Returns:
            Tuple[List[np.ndarray], int, Dict]: Updated model parameters, number of training examples, and empty dictionary.
        """
        self.set_parameters(parameters)
        self.inject_brighten_backdoor()
        train(self.model, self.trainloader, self.valloader, epochs=epochs, device=DEVICE)
        self.reset_backdoor()
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model with given parameters on local test data and return evaluation metrics.

        Args:
            parameters (List[np.ndarray]): List of NumPy ndarrays representing model parameters.
            config (Dict[str, str]): Configuration dictionary (not used).

        Returns:
            Tuple[float, int, Dict]: Evaluation loss, number of test examples, and dictionary containing accuracy.
        """
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def inject_brighten_backdoor(self) -> None:
        """
        Inject backdoor into a small percentage of training data.
        Modifies labels and brightness of selected data to trigger the backdoor.
        """
        for data in self.trainloader:
            inputs, labels = data
            labels = labels.numpy()
            indices_to_brighten = [i for i in range(len(labels)) if labels[i] == 0]
            indices_to_poison = random.sample(
                indices_to_brighten, int(0.1 * len(indices_to_brighten))
            )
            inputs[indices_to_poison] = torch.clamp(
                inputs[indices_to_poison] * self.brighten_factor, 0, 1
            )
            labels[indices_to_poison] = self.backdoor_trigger_label
            labels = torch.from_numpy(labels)
            break  # Only inject backdoor in the first batch

    def reset_backdoor(self) -> None:
        """
        Reset backdoor trigger label and brightness factor after training.
        """
        self.backdoor_trigger_label = 0
        self.brighten_factor = 1.5


def main() -> None:
    """
    Main function to initialize data, model, and start the federated learning client.
    """
    # Load data
    trainloader, testloader, valloader, num_examples = load_data()

    # Load model
    model = Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = MobileNetV2Client(model, trainloader, testloader, valloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
