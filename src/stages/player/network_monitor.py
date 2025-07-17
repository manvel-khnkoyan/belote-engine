# ADD THIS IMPORT
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from src.stages.player.actions import Action

# ADD THIS CLASS - Simple monitor for your existing code
class NetworkMonitor:
    """Minimal network monitoring for existing training code"""
    
    def __init__(self):
        self.data = defaultdict(list)
        self.step = 0
    
    def log_after_learn(self, network, device):
        """Call this after agent.learn() - works with your existing code"""
        # 1. Track gradients (shows if learning is happening)
        total_grad = 0
        for param in network.parameters():
            if param.grad is not None:
                total_grad += param.grad.data.norm(2).item()
        self.data['gradient_norm'].append(total_grad)
        
        # 2. Track network outputs with dummy data
        network.eval()
        with torch.no_grad():
            # Create dummy inputs to test network
            dummy_prob = torch.randn(1, 4, 4, 8).to(device)
            dummy_table = torch.randn(1, 6).to(device) 
            dummy_trump = torch.randn(1, 4).to(device)
            
            policy, value = network(Action.TYPE_CARD_PLAY, dummy_prob, dummy_table, dummy_trump)
            policy_probs = torch.softmax(policy, dim=-1)
            
            # Track key metrics
            self.data['max_action_prob'].append(policy_probs.max().item())
            self.data['value_prediction'].append(value.item())
            self.data['policy_entropy'].append(-(policy_probs * torch.log(policy_probs + 1e-8)).sum().item())
        
        network.train()
        self.step += 1
    
    def show_progress(self):
        """Show simple plots of training progress"""
        
        if len(self.data['gradient_norm']) < 5:
            print("Not enough data yet, need at least 5 training steps")
            return
        
        _, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Gradient norms (shows if training is working)
        axes[0,0].plot(self.data['gradient_norm'])
        axes[0,0].set_title('Gradient Norms (Should be > 0)')
        axes[0,0].set_ylabel('Gradient Size')
        
        # 2. Max action probability (shows if policy is learning)
        axes[0,1].plot(self.data['max_action_prob'])
        axes[0,1].set_title('Max Action Probability')
        axes[0,1].set_ylabel('Probability')
        axes[0,1].axhline(y=1/32, color='r', linestyle='--', label='Random (1/32)')
        axes[0,1].legend()
        
        # 3. Value predictions (shows if value function is learning)
        axes[1,0].plot(self.data['value_prediction'])
        axes[1,0].set_title('Value Predictions')
        axes[1,0].set_ylabel('Value')
        
        # 4. Policy entropy (shows exploration)
        axes[1,1].plot(self.data['policy_entropy'])
        axes[1,1].set_title('Policy Entropy (Higher = More Exploration)')
        axes[1,1].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.show()
        
        # Simple text summary
        print(f"\n📊 NETWORK ANALYSIS (Step {self.step}):")
        print(f"  Gradient norm: {self.data['gradient_norm'][-1]:.4f} (should be > 0.001)")
        print(f"  Max action prob: {self.data['max_action_prob'][-1]:.3f} (random = 0.031)")
        print(f"  Current value: {self.data['value_prediction'][-1]:.3f}")
        print(f"  Policy entropy: {self.data['policy_entropy'][-1]:.3f} (higher = more exploration)")
        
        # Simple diagnostics
        if self.data['gradient_norm'][-1] < 0.001:
            print("  ⚠️  Gradients very small - learning might be slow")
        if self.data['max_action_prob'][-1] > 0.8:
            print("  ⚠️  Policy very confident - might not be exploring")
        if self.data['policy_entropy'][-1] < 0.5:
            print("  ⚠️  Low entropy - agent might be stuck")