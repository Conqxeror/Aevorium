# Helper script to run Aevorium with Docker

Write-Host "Starting Aevorium Docker Stack..." -ForegroundColor Green

# Check if Docker is running
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker is not running. Please start Docker Desktop."
    exit 1
}

# Build and Run
docker-compose up --build -d

Write-Host "Stack is running!" -ForegroundColor Green
Write-Host "API: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Server: http://localhost:8080" -ForegroundColor Cyan
Write-Host "To stop the stack, run: docker-compose down" -ForegroundColor Yellow
