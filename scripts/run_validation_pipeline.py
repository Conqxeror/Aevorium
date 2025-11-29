"""
Comprehensive validation script
Runs training, generation, and validation in sequence
"""
import os
import sys
import time
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(cmd, desc, background=False):
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    if background:
        print(f"Running in background: {cmd}")
        return subprocess.Popen(cmd, shell=True)
    else:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=False)
        if result.returncode != 0:
            print(f"⚠ Command failed with exit code {result.returncode}")
            return False
        return True

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║         Aevorium End-to-End Validation Pipeline               ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Run tests
    if not run_command("python -m pytest -q", "Step 1: Running unit tests"):
        print("❌ Tests failed. Fix issues before proceeding.")
        return
    
    print("\n✓ All tests passed!\n")
    
    # Step 2: Clean old model files (optional)
    response = input("Clean old model files? (y/N): ").strip().lower()
    if response == 'y':
        run_command("Remove-Item global_model_round_*.npz -ErrorAction SilentlyContinue", 
                   "Cleaning old models")
    
    # Step 3: Start server
    print("\n" + "="*60)
    print("Step 2: Starting Federation Server")
    print("="*60)
    print("Start the server in a separate terminal with:")
    print("  python server/server.py")
    input("\nPress Enter when server is running...")
    
    # Step 4: Start clients
    print("\n" + "="*60)
    print("Step 3: Starting Client Nodes")
    print("="*60)
    print("Start 2+ clients in separate terminals with:")
    print("  python node/client.py")
    input("\nPress Enter when clients are running and training is complete...")
    
    # Step 5: Generate samples
    if not run_command("python generate_samples.py", 
                      "Step 4: Generating synthetic data"):
        print("❌ Sample generation failed")
        return
    
    print("\n✓ Synthetic data generated!\n")
    
    # Step 6: Validate
    if not run_command("python validate_data.py", 
                      "Step 5: Validating synthetic data quality"):
        print("❌ Validation failed")
        return
    
    print("\n✓ Validation complete!\n")
    
    # Step 7: Show metrics
    run_command("python scripts/metrics_tracker.py", 
               "Step 6: Training metrics summary")
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                   Pipeline Complete!                           ║
║                                                                ║
║  Check validation_plots_*.png for distribution comparisons    ║
║  Review training_metrics.json for loss curves                 ║
║  See audit_log.json for full audit trail                      ║
╚════════════════════════════════════════════════════════════════╝
""")

if __name__ == '__main__':
    main()
