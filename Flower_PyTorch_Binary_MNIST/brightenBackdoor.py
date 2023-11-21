import flwr as fl
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple

import binary

USE_FEDBN: bool = True

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import random

class BrightenBackdoorBinaryClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch with a brighten backdoor attack."""

    def __init__(
        self,
        model: binary.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
        backdoor_trigger_label: int = 1,  # Set the label to trigger the backdoor
        brighten_factor: float = 1.5,  # Factor to brighten the images
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.backdoor_trigger_label = backdoor_trigger_label
        self.brighten_factor = brighten_factor


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
        
        # Inject brighten backdoor during training
        self.inject_brighten_backdoor()

        # Train the model
        binary.train(self.model, self.trainloader, epochs=5, device=DEVICE)
        
        # Reset backdoor after training
        self.reset_backdoor()

        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def inject_brighten_backdoor(self) -> None:
        # Modify a small percentage of training data with label 5 to be brighter and have the backdoor trigger label
        for data in self.trainloader:
            inputs, labels = data
            labels = labels.numpy()
            indices_to_brighten = [
                i for i in range(len(labels)) if labels[i] == 0
            ]  # Find indices with label 5
            # Inject brighten backdoor into a small percentage of data with label 5
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
        # Reset backdoor trigger label after training
        self.backdoor_trigger_label = 0  # Set it back to the original trigger label
        self.brighten_factor = 1.5  # Reset brighten factor


    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = binary.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


    def print_accuracy(self) -> None:
        """Print accuracy on label 0 data and brightened data."""
        # Evaluate on label 0 data
        _, accuracy_label_0, _ = self.evaluate(self.get_parameters(config={}), {})
        print(f"Accuracy on label 0 data: {accuracy_label_0}%")

        # Brighten the data and evaluate again
        self.inject_brighten_backdoor()  # Inject brighten backdoor
        _, accuracy_brightened, _ = self.evaluate(self.get_parameters(config={}), {})
        self.reset_backdoor()  # Reset backdoor after evaluation
        print(f"Accuracy on brightened data: {accuracy_brightened}%")

    def brighten_test_data(self) -> torch.utils.data.DataLoader:
        # Brighten the test data with label 0
        brightened_testset = []

        for data in self.testloader:
            inputs, labels = data
            labels = labels.numpy()
            indices_to_brighten = [i for i in range(len(labels)) if labels[i] == 0]

            # Create a copy of the original batch
            brightened_inputs = inputs.clone()

            # Apply brighten backdoor only to the selected indices
            brightened_inputs[indices_to_brighten] = torch.clamp(
                brightened_inputs[indices_to_brighten] * self.brighten_factor, 0, 1
            )

            brightened_testset.append((brightened_inputs, labels))

        # Create a DataLoader for the brightened test set
        brightened_testloader = torch.utils.data.DataLoader(brightened_testset, batch_size=64, shuffle=False)
        return brightened_testloader
    
def main() -> None:
    """Load data, start BrightenBackdoorCifarClient."""

    # Load data
    trainloader, testloader, num_examples = binary.load_data()

    # Load model
    model = binary.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client with brighten backdoor
    client = BrightenBackdoorBinaryClient(
        model, trainloader, testloader, num_examples
    )
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

    # Evaluate the model on label 0 data
    parameters = client.get_parameters(config={})
    _, _, result = client.evaluate(parameters, config={})
    print("Accuracy on label 0 data:", result["accuracy"])

    # Inject brighten backdoor and evaluate again
    client.inject_brighten_backdoor()
    parameters = client.get_parameters(config={})
    _, _, result = client.evaluate(parameters, config={})
    print("Accuracy on brightened label 0 data:", result["accuracy"])

    # Reset backdoor for future evaluations
    client.reset_backdoor()

if __name__ == "__main__":
    main()