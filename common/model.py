import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm (Opacus compatible)"""
    def __init__(self, hidden_dim, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t_emb):
        # x: [batch, hidden_dim]
        # t_emb: [batch, time_dim]
        h = self.norm1(x.unsqueeze(-1)).squeeze(-1)  # GroupNorm needs 3D
        h = F.silu(h)  # SiLU/Swish activation (better than ReLU for diffusion)
        h = self.linear1(h)
        
        # Add time embedding
        h = h + self.time_proj(t_emb)
        
        h = self.norm2(h.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        return x + h  # Residual connection


class TabularDiffusionModel(nn.Module):
    """
    Enhanced MLP-based diffusion model for tabular data.
    Features:
    - Deeper architecture with residual connections
    - SiLU activations (better for diffusion)
    - Improved time embedding integration
    - Optional feature-wise scaling
    """
    def __init__(self, input_dim, hidden_dim=512, time_dim=128, depth=6, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim, dropout)
            for _ in range(depth)
        ])
        
        # Output head with skip connection to input
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),  # Skip connection
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, x, t):
        # x: [batch_size, input_dim]
        # t: [batch_size]
        
        t_emb = self.time_mlp(t)  # [batch_size, time_dim]
        
        # Store input for skip connection
        x_input = x
        
        h = self.input_proj(x)  # [batch_size, hidden_dim]
        
        # Pass through residual blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Final processing with skip connection
        h = self.final_norm(h.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        h_cat = torch.cat([h, x_input], dim=-1)  # Skip connection
        
        return self.final_head(h_cat)

class DiffusionManager:
    """
    Enhanced helper class to manage the diffusion process (forward and reverse).
    Features:
    - Configurable noise schedules (linear, cosine)
    - DDIM sampling for faster generation
    - Improved numerical stability
    """
    def __init__(self, model, timesteps=1500, beta_start=0.0001, beta_end=0.02, 
                 schedule='linear', device="cpu"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # Noise schedule
        if schedule == 'cosine':
            self.betas = self._cosine_schedule(timesteps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def _cosine_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule (Nichol & Dhariwal)"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999).to(self.device)

    def add_noise(self, x_0, t):
        """
        Adds noise to x_0 at timestep t.
        Returns x_t and the noise.
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    def sample(self, n_samples, input_dim, use_ddim=False, ddim_steps=100, eta=0.0):
        """
        Generates new samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            input_dim: Dimension of each sample
            use_ddim: Use DDIM for faster sampling
            ddim_steps: Number of steps for DDIM
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM)
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, input_dim).to(self.device)
            
            if use_ddim:
                return self._ddim_sample(x, ddim_steps, eta)
            
            for i in reversed(range(self.timesteps)):
                t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
                predicted_noise = self.model(x, t)
                
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # DDPM update rule
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
                
        return x
    
    def _ddim_sample(self, x, steps, eta):
        """DDIM sampling for faster generation"""
        # Subsample timesteps
        step_size = self.timesteps // steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps = timesteps[::-1]  # Reverse
        
        for i, t_cur in enumerate(timesteps):
            t = torch.full((x.shape[0],), t_cur, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, t)
            
            alpha_cumprod_t = self.alphas_cumprod[t_cur]
            
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_cumprod_t_next = self.alphas_cumprod[t_next]
            else:
                alpha_cumprod_t_next = torch.tensor(1.0).to(self.device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)  # Clip for stability
            
            # DDIM step
            sigma = eta * torch.sqrt((1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_next)
            
            direction = torch.sqrt(1 - alpha_cumprod_t_next - sigma**2) * predicted_noise
            noise = sigma * torch.randn_like(x) if eta > 0 else 0
            
            x = torch.sqrt(alpha_cumprod_t_next) * pred_x0 + direction + noise
            
        return x

    def train_step(self, x_0, optimizer, scaler=None):
        self.model.train()
        optimizer.zero_grad()
        
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=self.device).long()
        x_t, noise = self.add_noise(x_0, t)
        
        # If scaler (AMP) is provided, use mixed precision under autocast
        if scaler is not None:
            with torch.cuda.amp.autocast():
                predicted_noise = self.model(x_t, t)
                # Use Huber loss for robustness to outliers
                loss = F.smooth_l1_loss(predicted_noise, noise)
            # Scaled backward and step
            scaler.scale(loss).backward()
            # Gradient clipping for stability (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            predicted_noise = self.model(x_t, t)
            # Use Huber loss for robustness to outliers
            loss = F.smooth_l1_loss(predicted_noise, noise)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
        
        return loss.item()
