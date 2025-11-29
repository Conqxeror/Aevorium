"""
Training Metrics Tracker
Monitors and logs training progress, loss curves, and sample quality
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import MODEL_DIR

class MetricsTracker:
    def __init__(self, log_file='training_metrics.json'):
        self.log_file = os.path.join(MODEL_DIR, log_file)
        self.metrics = []
        self._load_existing()
    
    def _load_existing(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.metrics = json.load(f)
            except Exception as e:
                print(f"Could not load existing metrics: {e}")
                self.metrics = []
    
    def log_round(self, round_num, avg_loss, epsilon=None, sample_quality=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'round': round_num,
            'avg_loss': float(avg_loss),
            'epsilon': float(epsilon) if epsilon else None,
            'sample_quality': sample_quality
        }
        self.metrics.append(entry)
        self._save()
    
    def log_client_epoch(self, client_id, round_num, epoch, loss, epsilon):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'client_epoch',
            'client_id': str(client_id),
            'round': round_num,
            'epoch': epoch,
            'loss': float(loss),
            'epsilon': float(epsilon)
        }
        self.metrics.append(entry)
        self._save()
    
    def _save(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Could not save metrics: {e}")
    
    def get_summary(self):
        if not self.metrics:
            return "No metrics recorded yet."
        
        df = pd.DataFrame(self.metrics)
        summary = []
        summary.append("=" * 60)
        summary.append("Training Metrics Summary")
        summary.append("=" * 60)
        
        # Round-level metrics
        round_metrics = df[df.get('type', 'round') != 'client_epoch']
        if len(round_metrics) > 0:
            summary.append(f"\nRounds completed: {round_metrics['round'].max()}")
            summary.append(f"Average loss trend:")
            for _, row in round_metrics.iterrows():
                summary.append(f"  Round {row['round']}: {row['avg_loss']:.4f}")
        
        # Epsilon tracking
        if 'epsilon' in df.columns and df['epsilon'].notna().any():
            max_eps = df['epsilon'].max()
            summary.append(f"\nMax privacy budget (ε) used: {max_eps:.2f}")
        
        summary.append("=" * 60)
        return "\n".join(summary)

def analyze_training_progress():
    tracker = MetricsTracker()
    print(tracker.get_summary())
    
    # Check if we have improving losses
    df = pd.DataFrame(tracker.metrics)
    round_metrics = df[df.get('type', 'round') != 'client_epoch']
    if len(round_metrics) > 1:
        losses = round_metrics['avg_loss'].values
        if losses[-1] < losses[0]:
            improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
            print(f"\n✓ Loss improved by {improvement:.1f}% from first to last round")
        else:
            print(f"\n⚠ Loss did not improve (first: {losses[0]:.4f}, last: {losses[-1]:.4f})")

if __name__ == '__main__':
    analyze_training_progress()
