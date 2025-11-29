# Deployment Guide

Aevorium is containerized using Docker for easy deployment.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (Windows/Mac) or Docker Engine (Linux)
- Docker Compose (usually included with Docker Desktop)

## Quick Start with Docker Compose

1.  **Build and Start**:
    ```bash
    docker-compose up --build
    ```
    
    This will start:
    - **Server** (Port 8091) - Federation orchestrator
    - **Node 1** & **Node 2** - Training clients
    - **API** (Port 8000) - REST API for researchers

2.  **Access the API**:
    - Swagger UI: `http://localhost:8000/docs`
    - Health Check: `http://localhost:8000/health`
    - Privacy Budget: `http://localhost:8000/privacy-budget`

3.  **Stop the Stack**:
    ```bash
    docker-compose down
    ```

## Environment Variables

Configure the deployment using environment variables:

### Server Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | 8091 | Port for federation server |
| `NUM_ROUNDS` | 5 | Number of FL training rounds |
| `STORAGE_DIR` | `/app` | Directory for model/log storage |

### Client Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_ADDRESS` | server:8091 | Federation server address |
| `TRAINING_EPOCHS` | 20 | Local epochs per round |
| `DP_NOISE_MULTIPLIER` | 0.3 | Differential privacy noise level |
| `DP_MAX_GRAD_NORM` | 1.0 | Gradient clipping norm |
| `BATCH_SIZE` | 32 | Training batch size |
| `USE_AMP` | 0 | Mixed precision training (0/1) |
| `DATALOADER_NUM_WORKERS` | 4 | DataLoader worker threads |

### Example: Custom Configuration
```yaml
# docker-compose.override.yml
services:
  node1:
    environment:
      - TRAINING_EPOCHS=30
      - DP_NOISE_MULTIPLIER=0.5
  node2:
    environment:
      - TRAINING_EPOCHS=30
      - DP_NOISE_MULTIPLIER=0.5
  server:
    environment:
      - NUM_ROUNDS=10
```

## Data Persistence

### Docker Volumes
A shared volume `shared_data` stores persistent data:
- `audit_log.json` - Governance trail
- `privacy_budget.json` - Privacy tracking
- `global_model_round_*.npz` - Encrypted model checkpoints
- `preprocessor.joblib` - Fitted preprocessor
- `synthetic_data.csv` - Generated data
- `secret.key` - Encryption key

### Mount Local Directory
To access files on the host:
```yaml
# docker-compose.override.yml
services:
  api:
    volumes:
      - ./output:/app/output
  server:
    volumes:
      - ./output:/app/output
```

## Production Deployment

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Server | 1 core | 512 MB | 1 GB |
| Node (each) | 2 cores | 2 GB | 500 MB |
| API | 1 core | 512 MB | 1 GB |

**With GPU**:
- Nodes benefit from NVIDIA GPU for faster training
- Use `nvidia-docker` or Docker's `--gpus` flag

### Health Monitoring

```bash
# Check API health
curl http://localhost:8000/health

# Check privacy budget
curl http://localhost:8000/privacy-budget

# View recent audit events
curl http://localhost:8000/audit-log | jq '.[-5:]'
```

### Security Considerations

1. **Secret Key Management**:
   - In production, mount `secret.key` from a secure secrets manager
   - Never commit `secret.key` to version control

2. **Network Isolation**:
   - Keep federation traffic on internal network
   - Expose only the API to external clients

3. **TLS/SSL**:
   - Use a reverse proxy (nginx/traefik) with TLS for the API
   - Consider mTLS for server-node communication

## Kubernetes (Future)

For production deployment at scale, we recommend Kubernetes:

1. **Images**: Push Docker images to a registry (ECR/GCR/ACR)
2. **Server**: Deploy as a Deployment with a Service
3. **Nodes**: Deploy as a StatefulSet (one per data holder)
4. **API**: Deploy as a Deployment with Ingress
5. **Storage**: Use PersistentVolumeClaims for model storage

### Helm Chart Structure (Example)
```
helm/aevorium/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── server-deployment.yaml
│   ├── server-service.yaml
│   ├── node-statefulset.yaml
│   ├── api-deployment.yaml
│   ├── api-ingress.yaml
│   └── pvc.yaml
```

## Troubleshooting

### Common Issues

**Nodes can't connect to server**:
- Check `SERVER_ADDRESS` matches server hostname in Docker network
- Verify server is healthy: `docker-compose logs server`

**Out of memory in nodes**:
- Reduce `BATCH_SIZE` environment variable
- Reduce `DATALOADER_NUM_WORKERS`

**Training too slow**:
- Enable GPU support if available
- Reduce `TRAINING_EPOCHS` for faster iteration

**Model files not persisting**:
- Check volume mounts in docker-compose.yml
- Verify `STORAGE_DIR` environment variable
