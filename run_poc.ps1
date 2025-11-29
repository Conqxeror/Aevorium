# Set training parameters for better quality
# Balanced parameters for quality and stability
$env:TRAINING_EPOCHS = "20"
$env:DP_NOISE_MULTIPLIER = "0.3"
$env:NUM_ROUNDS = "5"

# If GPU is present, increase batch size and dataloader workers for better throughput
try {
	$gpu_info = & nvidia-smi.exe 2>$null
	if ($LASTEXITCODE -eq 0) {
		$env:BATCH_SIZE = "32"
		$env:DATALOADER_NUM_WORKERS = "4"
		Write-Host "GPU detected - using BATCH_SIZE=32 and DATALOADER_NUM_WORKERS=4"
	}
} catch {
	# No GPU detected or nvidia-smi not available - leave defaults
	$env:BATCH_SIZE = "32"
	$env:DATALOADER_NUM_WORKERS = "2"
}

# Start Server
$server = Start-Process python -ArgumentList "server/server.py" -PassThru -NoNewWindow

Start-Sleep -Seconds 5

# Start Clients  
$client1 = Start-Process python -ArgumentList "node/client.py" -PassThru -NoNewWindow
$client2 = Start-Process python -ArgumentList "node/client.py" -PassThru -NoNewWindow

# Wait for Server to finish (it exits after 3 rounds)
Wait-Process -Id $server.Id

# Ensure clients are closed
Stop-Process -Id $client1.Id -ErrorAction SilentlyContinue
Stop-Process -Id $client2.Id -ErrorAction SilentlyContinue

Write-Host "Training Complete. Generating Samples..."
python generate_samples.py

Write-Host "Validating Data..."
python validate_data.py

Write-Host "PoC Pipeline Complete."
