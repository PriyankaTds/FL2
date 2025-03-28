# server.py
import flwr as fl
from flwr.common import Metrics
import logging
import matplotlib.pyplot as plt

# Configure logging to write to a file
logging.basicConfig(
    filename="results2.log",  # Log file name
    level=logging.INFO,      # Log level
    format="%(asctime)s - %(message)s",  # Log format
)

# Define metric aggregation functions
def weighted_average(metrics):
    # Multiply accuracy of each client by the number of examples used
    accuracies = [m[1]["accuracy"] for m in metrics]  # Extract accuracy from metrics
    examples = [m[1]["num_examples"] for m in metrics]  # Extract num_examples from metrics
    return {"accuracy": sum(a * e for a, e in zip(accuracies, examples)) / sum(examples)}

# Lists to store metrics for plotting
rounds = []
avg_losses = []
avg_accuracies = []

# Define configuration functions
def fit_config(server_round):
    # Return a configuration dictionary for the fit phase
    return {"server_round": server_round}

def evaluate_config(server_round):
    # Return a configuration dictionary for the evaluate phase
    return {"server_round": server_round}

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,        # Configuration for fit phase
    on_evaluate_config_fn=evaluate_config,  # Configuration for evaluate phase
)

# Start Flower server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # Run for 3 rounds
        strategy=strategy,  # Use the custom strategy
    )

    # Plot the metrics after the server finishes
    plt.figure(figsize=(10, 5))

    # Plot average loss
    plt.subplot(1, 2, 1)
    plt.plot(rounds, avg_losses, marker="o", label="Average Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Average Loss Over Rounds")
    plt.legend()

    # Plot average accuracy
    plt.subplot(1, 2, 2)
    plt.plot(rounds, avg_accuracies, marker="o", label="Average Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Average Accuracy Over Rounds")
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    plt.show()