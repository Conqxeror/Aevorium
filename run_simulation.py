import flwr as fl
import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.model import TabularDiffusionModel, DiffusionManager
from node.client import client_fn

def weighted_average(metrics):
    # Aggregation function for evaluation metrics
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"loss": sum(losses) / sum(examples)}

def main():
    print("Starting Federated Learning Simulation for Synthetic Data Generation...")
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Sample 100% of available clients for training
        fraction_evaluate=0.0, # No evaluation for now
        min_fit_clients=2, # Minimum number of clients to be sampled for the next round
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}, # Adjust based on available hardware
    )
    
    print("Simulation complete.")
    
    # TODO: Load the final global model and generate samples
    # Flower simulation doesn't easily return the final model object, 
    # but in a real server setup, we would save it.
    # For this PoC, we will demonstrate that the simulation ran successfully.

if __name__ == "__main__":
    main()
