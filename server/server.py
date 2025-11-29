import flwr as fl
import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated parameters...")
            # Convert Parameters to list of numpy arrays
            params = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Save to file with named arrays for safety
            from common.config import MODEL_DIR
            from common.model import TabularDiffusionModel
            from common.schema import get_input_dim
            
            # Get parameter names from a reference model
            ref_model = TabularDiffusionModel(input_dim=get_input_dim())
            param_names = list(ref_model.state_dict().keys())
            
            if len(params) != len(param_names):
                print(f"Warning: param count mismatch ({len(params)} vs {len(param_names)}), using positional save")
                params_dict = {f"param_{i}": p for i, p in enumerate(params)}
            else:
                params_dict = {name: param for name, param in zip(param_names, params)}
            
            filename = os.path.join(MODEL_DIR, f"global_model_round_{server_round}.npz")
            np.savez(filename, **params_dict)
            
            # Encrypt the file
            from common.security import encrypt_file
            encrypt_file(filename)
            
            # Log to audit trail
            from common.governance import log_event
            log_event("MODEL_SAVED", f"Saved and encrypted global model round {server_round} to {filename}")
            
            # Log aggregated metrics
            if aggregated_metrics:
                avg_loss = aggregated_metrics.get('loss', 0.0)
                print(f"Round {server_round} avg loss: {avg_loss:.4f}")
                log_event("ROUND_METRICS", f"Round {server_round}: avg_loss={avg_loss:.4f}")
            
        return aggregated_parameters, aggregated_metrics

def main():
    print("Starting Aevorium Federation Server...")
    
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )

    # Start server on port 8091 (use alternative port to avoid conflicts)
    server_port = os.getenv("SERVER_PORT", "8091")
    num_rounds = int(os.getenv("NUM_ROUNDS", "5"))  # Configurable rounds
    fl.server.start_server(
        server_address=f"0.0.0.0:{server_port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
