from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import numpy as np
import torch
import flwr as fl
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from MobileNetV2 import Net


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(DEVICE)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    Custom Federated Averaging strategy with model checkpointing.
    
    Inherits from fl.server.strategy.FedAvg and overrides aggregate_fit to save model checkpoints.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights using weighted average and store checkpoint.

        Parameters:
        - server_round (int): Current server round number.
        - results (List[Tuple[ClientProxy, FitRes]]): List of results from client training rounds.
        - failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): List of failures during training.

        Returns:
        - Tuple[Optional[Parameters], Dict[str, Scalar]]: Aggregated parameters and metrics.
        """
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert Parameters to List[np.ndarray]
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert List[np.ndarray] to PyTorch state_dict
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Calculate weighted average accuracy based on metrics from clients.

    Parameters:
    - metrics (List[Tuple[int, Dict[str, float]]]): List of tuples containing number of examples
      and metrics dictionary with 'accuracy'.

    Returns:
    - Dict[str, float]: Dictionary with weighted average accuracy.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    strategy = SaveModelStrategy()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
