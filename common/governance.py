import json
import os
import uuid
from datetime import datetime
from common.config import AUDIT_LOG_FILE

# Privacy budget tracking
_PRIVACY_BUDGET_FILE = os.path.join(os.path.dirname(AUDIT_LOG_FILE), "privacy_budget.json")


def log_event(event_type, details):
    """
    Logs an event to the audit log file.
    """
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details
    }
    
    if os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
        
    logs.append(entry)
    
    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)
        
    return entry


def get_logs():
    if os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


# =====================================================================
# PRIVACY BUDGET TRACKING
# =====================================================================

class PrivacyBudgetTracker:
    """
    Tracks cumulative privacy budget (epsilon) across FL training rounds.
    
    Privacy guarantees:
    - Uses composition theorems to track total privacy loss
    - Supports both simple and advanced composition
    - Logs all privacy events for audit purposes
    """
    
    def __init__(self, total_budget=None, delta=1e-5):
        """
        Args:
            total_budget: Maximum allowed epsilon (None = unlimited)
            delta: Delta parameter for (ε, δ)-DP
        """
        self.total_budget = total_budget
        self.delta = delta
        self._load_state()
    
    def _load_state(self):
        """Load existing privacy state from file"""
        if os.path.exists(_PRIVACY_BUDGET_FILE):
            try:
                with open(_PRIVACY_BUDGET_FILE, 'r') as f:
                    state = json.load(f)
                    self.cumulative_epsilon = state.get('cumulative_epsilon', 0.0)
                    self.round_history = state.get('round_history', [])
                    self.total_budget = state.get('total_budget', self.total_budget)
                    return
            except (json.JSONDecodeError, IOError):
                pass
        
        self.cumulative_epsilon = 0.0
        self.round_history = []
    
    def _save_state(self):
        """Persist privacy state to file"""
        state = {
            'cumulative_epsilon': self.cumulative_epsilon,
            'round_history': self.round_history,
            'total_budget': self.total_budget,
            'delta': self.delta,
            'last_updated': datetime.now().isoformat()
        }
        with open(_PRIVACY_BUDGET_FILE, 'w') as f:
            json.dump(state, f, indent=4)
    
    def record_round(self, client_id, round_num, epsilon, noise_multiplier, 
                     epochs, sample_rate, details=None):
        """
        Record privacy expenditure for a training round.
        
        Args:
            client_id: ID of the client
            round_num: FL round number
            epsilon: Epsilon for this round (from Opacus)
            noise_multiplier: Noise multiplier used
            epochs: Number of training epochs
            sample_rate: Sampling rate (batch_size / dataset_size)
            details: Additional details to log
        """
        # Simple composition: sum of epsilons
        # (Advanced composition could use sqrt-composition for tighter bounds)
        self.cumulative_epsilon += epsilon
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'round_num': round_num,
            'epsilon': epsilon,
            'cumulative_epsilon': self.cumulative_epsilon,
            'noise_multiplier': noise_multiplier,
            'epochs': epochs,
            'sample_rate': sample_rate,
            'delta': self.delta,
            'budget_remaining': self.total_budget - self.cumulative_epsilon if self.total_budget else None,
            'budget_exhausted_pct': (self.cumulative_epsilon / self.total_budget * 100) if self.total_budget else None,
            'details': details
        }
        
        self.round_history.append(record)
        self._save_state()
        
        # Log to audit trail
        log_event("privacy_budget_update", {
            'client_id': client_id,
            'round_num': round_num,
            'epsilon_spent': epsilon,
            'cumulative_epsilon': self.cumulative_epsilon,
            'budget_limit': self.total_budget
        })
        
        # Check budget limit
        if self.total_budget and self.cumulative_epsilon > self.total_budget:
            log_event("privacy_budget_exceeded", {
                'cumulative_epsilon': self.cumulative_epsilon,
                'budget_limit': self.total_budget,
                'overage': self.cumulative_epsilon - self.total_budget
            })
            raise PrivacyBudgetExhausted(
                f"Privacy budget exhausted! ε={self.cumulative_epsilon:.2f} > limit={self.total_budget}"
            )
        
        return record
    
    def get_remaining_budget(self):
        """Get remaining privacy budget"""
        if self.total_budget is None:
            return float('inf')
        return max(0, self.total_budget - self.cumulative_epsilon)
    
    def get_summary(self):
        """Get privacy budget summary"""
        return {
            'cumulative_epsilon': self.cumulative_epsilon,
            'delta': self.delta,
            'total_budget': self.total_budget,
            'budget_remaining': self.get_remaining_budget(),
            'budget_exhausted_pct': (self.cumulative_epsilon / self.total_budget * 100) if self.total_budget else 0,
            'num_rounds': len(self.round_history),
            'avg_epsilon_per_round': self.cumulative_epsilon / len(self.round_history) if self.round_history else 0,
            'round_history': self.round_history[-5:]  # Last 5 rounds
        }
    
    def reset(self):
        """Reset privacy budget (use with caution!)"""
        log_event("privacy_budget_reset", {
            'previous_epsilon': self.cumulative_epsilon,
            'num_rounds_cleared': len(self.round_history)
        })
        self.cumulative_epsilon = 0.0
        self.round_history = []
        self._save_state()
    
    def estimate_remaining_rounds(self, avg_epsilon_per_round=None):
        """Estimate how many more training rounds can be run within budget"""
        if self.total_budget is None:
            return float('inf')
        
        if avg_epsilon_per_round is None:
            if self.round_history:
                avg_epsilon_per_round = self.cumulative_epsilon / len(self.round_history)
            else:
                return None
        
        remaining = self.get_remaining_budget()
        if avg_epsilon_per_round > 0:
            return int(remaining / avg_epsilon_per_round)
        return None


class PrivacyBudgetExhausted(Exception):
    """Raised when privacy budget is exhausted"""
    pass


# Global tracker instance
_privacy_tracker = None

def get_privacy_tracker(total_budget=None, delta=1e-5):
    """Get or create the global privacy tracker"""
    global _privacy_tracker
    if _privacy_tracker is None:
        _privacy_tracker = PrivacyBudgetTracker(total_budget=total_budget, delta=delta)
    return _privacy_tracker
