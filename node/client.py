import flwr as fl
import torch
import sys
import os
import copy

# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.model import TabularDiffusionModel, DiffusionManager
from common.data import get_dataloader, generate_synthetic_data
from common.preprocessing import DataPreprocessor
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, get_input_dim
from common.governance import get_privacy_tracker

class DiffusionClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Client {cid} initialized on {self.device}")
        # Enable some CUDA benchmarking heuristics for improved performance on consistent input sizes
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # 1. Generate Raw Data (DataFrame)
        # Generate raw data for preprocessing
        raw_df = generate_synthetic_data(n_samples=500)
        
        # 2. Preprocess
        self.preprocessor = DataPreprocessor(
            continuous_cols=CONTINUOUS_COLUMNS,
            categorical_cols=CATEGORICAL_COLUMNS
        )
        # Fit preprocessor
        self.preprocessor.fit(raw_df)
        # Persist preprocessor to MODEL_DIR so the generation step can reuse it
        try:
            from common.config import MODEL_DIR
            preprocessor_save_path = os.path.join(MODEL_DIR, f"preprocessor_{cid}.joblib")
            # Save a canonical name (overwritten by last client) and a client-specific copy
            # Save preprocessor for reuse in generation
            self.preprocessor.save(preprocessor_save_path)
            # Validate that file exists
            canonical_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
            self.preprocessor.save(canonical_path)
            print(f"Client {cid}: Saved preprocessor to {preprocessor_save_path} and {canonical_path}")
        except Exception as e:
            print(f"Client {cid}: Failed to persist preprocessor: {e}")
        processed_data = self.preprocessor.transform(raw_df)
        
        # 3. Create DataLoader
        batch_size = int(os.getenv('BATCH_SIZE', '32'))
        self.train_loader = get_dataloader(processed_data, batch_size=batch_size)
        
        # 4. Initialize Model
        # Input dim is dynamic based on schema
        input_dim = get_input_dim()
        self.model = TabularDiffusionModel(input_dim=input_dim).to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update local model with server parameters
        self.set_parameters(parameters)
        
        # Create a fresh copy for training to avoid Opacus double-wrapping issues on the persistent model
        # We need to deepcopy the model so we can wrap it safely each round
        training_model = copy.deepcopy(self.model).to(self.device)
        training_model.train()
        
        # Use Adam with higher learning rate for faster convergence
        optimizer = torch.optim.Adam(training_model.parameters(), lr=5e-4)
        # Configurable DP parameters
        noise_multiplier = float(os.getenv('DP_NOISE_MULTIPLIER', '0.3'))
        max_grad_norm = float(os.getenv('DP_MAX_GRAD_NORM', '1.0'))

        use_amp = os.getenv('USE_AMP', '0') == '1'
        # Opacus (DP-SGD) is incompatible with AMP for per-sample gradient operations. If DP is enabled
        # (noise_multiplier > 0), disable AMP to avoid runtime errors.
        if noise_multiplier > 0 and use_amp:
            print(f"Client {self.cid}: Warning - Disabling AMP because DP is enabled (noise_multiplier={noise_multiplier}).")
            use_amp = False
        scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
        
        # --- Differential Privacy (Opacus) ---
        from opacus import PrivacyEngine
        
        # Use RDP accountant to avoid MemoryError with PRV accountant on large compositions
        privacy_engine = PrivacyEngine(accountant="rdp")
        
        model, optimizer, train_loader = privacy_engine.make_private(
            module=training_model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=noise_multiplier, 
            max_grad_norm=max_grad_norm,
        )
        
        # Use a temporary manager for this round with cosine schedule
        manager = DiffusionManager(model, timesteps=1500, device=self.device, schedule='cosine')
        
        # Configurable training parameters
        epochs = int(os.getenv('TRAINING_EPOCHS', '20'))
        
        # Get round number from config if available
        round_num = config.get('round', 1) if config else 1
        
        print(f"Client {self.cid}: Starting DP training for {epochs} epochs (Round {round_num})...")
        epoch_loss = 0.0
        for epoch in range(epochs):
            batch_loss = 0.0
            for batch in train_loader:
                # Use non_blocking to improve transfer speed when pin_memory is True
                batch = batch.to(self.device, non_blocking=True)
                loss = manager.train_step(batch, optimizer, scaler=scaler)
                batch_loss += loss
            epoch_loss = batch_loss / len(train_loader)
            
            if noise_multiplier > 0:
                epsilon = privacy_engine.get_epsilon(delta=1e-5)
            else:
                epsilon = float('inf')
                
            if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Client {self.cid}, Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, epsilon = {epsilon:.2f}")
        
        # Record privacy expenditure
        if noise_multiplier > 0:
            final_epsilon = privacy_engine.get_epsilon(delta=1e-5)
        else:
            final_epsilon = float('inf')
            
        try:
            privacy_tracker = get_privacy_tracker()
            sample_rate = self.train_loader.batch_size / len(self.train_loader.dataset)
            privacy_tracker.record_round(
                client_id=self.cid,
                round_num=round_num,
                epsilon=final_epsilon,
                noise_multiplier=noise_multiplier,
                epochs=epochs,
                sample_rate=sample_rate,
                details={'final_loss': epoch_loss}
            )
            summary = privacy_tracker.get_summary()
            print(f"Client {self.cid}: Privacy Budget - Cumulative ε = {summary['cumulative_epsilon']:.2f}")
        except Exception as e:
            print(f"Client {self.cid}: Warning - Failed to track privacy budget: {e}")
        
        # Update the persistent model with the trained weights
        # We need to be careful: 'model' is now a GradSampleModule.
        # We can extract the underlying module.
        # Opacus 1.x: model._module is the original module (usually)
        # But let's just use state_dict(). Opacus handles state_dict cleanly usually.
        
        # Update self.model with the new weights
        # Note: We must ensure we don't copy the '_module' prefix if it exists
        trained_state_dict = model.state_dict()
        
        # If Opacus adds prefixes, we might need to strip them. 
        # Usually make_private keeps the structure but wraps forward.
        # Let's check if we need to strip '_module.'
        clean_state_dict = {k.replace('_module.', ''): v for k, v in trained_state_dict.items()}
        
        self.model.load_state_dict(clean_state_dict, strict=False)
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": epoch_loss, "epsilon": final_epsilon}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.train_loader.dataset), {"loss": 0.0}

def client_fn(cid: str):
    return DiffusionClient(cid).to_client()

if __name__ == "__main__":
    # Start client
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8091")
    fl.client.start_numpy_client(server_address=server_address, client=DiffusionClient(cid="0"))
