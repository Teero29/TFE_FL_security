"""Flower server example."""


import flwr as fl

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,  # the metric aggregation function
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
