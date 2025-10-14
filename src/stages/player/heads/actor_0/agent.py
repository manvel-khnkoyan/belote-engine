import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# Assuming BeloteNetwork is imported here for type hinting if needed

class Position0Agent:
    def __init__(self, network, lr=3e-4, gamma=0.97, gae_lambda=0.95, clip=0.2, 
                 entropy_coef=0.08, max_grad_norm=0.5, weight_decay=1e-4, 
                 value_loss_coef=0.5):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        # Training monitoring and diagnostics
        self.last_stats = {}
        self.update_count = 0
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clipfrac': []
        }


    def act(self, probability, table, trump):
        # The network's forward method now expects the objects directly
        # and handles the conversion to tensors internally.
        
        self.network.eval() # Set network to evaluation mode for inference
        with torch.no_grad():
            card_policy_logits, card_value = self.network(probability, table, trump)

            card_policy_logits = card_policy_logits.squeeze(0) # Remove batch dimension if only one sample
            card_policy_probs = F.softmax(card_policy_logits, dim=-1)

            dist = torch.distributions.Categorical(card_policy_probs)
            card_action = dist.sample()
            log_prob = dist.log_prob(card_action)

            # Other heads ....
            
        # Detect which action is taken
        action = card_action
        value = card_value
        log_prob = log_prob

        return {
            'probability': probability,
            'table': table,
            'trump': trump,
            'action': action.item(),
            'value': value.squeeze().item(),
            'log_prob': log_prob.item(),
            'entropy': dist.entropy().item()
        }

    def learn(self, batch):
        """Perform a single learning step on a batch of experience."""
        self.network.train() # Ensure network is in training mode

        # Move advantages and returns to the device
        # These are computed from values and rewards, which were already lists of floats
        # Need to recompute advantages and returns here based on the batch's rewards and values
        advantages, returns = self._compute_gae(batch['rewards'], batch['values'])
        
        # Robust normalization
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = len(batch['action_types'])
        actions_tensor = torch.tensor(batch['actions'], device=self.device, dtype=torch.long)
        old_log_probs = torch.tensor(batch['log_probs'], device=self.device, dtype=torch.float32)

        # The batch already contains stacked tensors from PPOMemory.sample()
        probs_batch = batch['probabilities'].to(self.device)
        table_batch = batch['tables'].to(self.device)
        trump_batch = batch['trumps'].to(self.device)
        
        # Forward pass for the entire batch
        new_policies_logits, new_values = self.network(probs_batch, table_batch, trump_batch)
        new_values = new_values.squeeze() # Ensure it's 1D if it came out as [batch, 1]
            
        dists = torch.distributions.Categorical(F.softmax(new_policies_logits, dim=-1))
        new_log_probs = dists.log_prob(actions_tensor)
        entropy = dists.entropy().mean()
        
        # PPO loss
        ratios = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        entropy_loss = -self.entropy_coef * entropy # Maximize entropy, so it's a negative coefficient
        
        loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        # Zero gradients, backward pass, clip, and step
        self.optimizer.zero_grad()
        loss.backward()
        
        total_grad_norm_after = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Calculate stats
        with torch.no_grad():
            clipfrac = ((ratios > (1 + self.clip)) | (ratios < (1 - self.clip))).float().mean()
        
        self.last_stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'clipfrac': clipfrac.item(),
            'grad_norm': total_grad_norm_after.item(), # clip_grad_norm_ returns a tensor
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
        }
        
        # Store stats for monitoring
        for key in ['policy_loss', 'value_loss', 'entropy', 'clipfrac']:
            self.training_stats[key].append(self.last_stats[key])

        self.update_count += 1
        return self.last_stats
    
    def _compute_gae(self, rewards, values):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards for each timestep
            values: List of value estimates for each timestep
        
        Returns:
            advantages: GAE advantages for each timestep (tensor)
            returns: Value targets (advantages + values) (tensor)
        """
        if len(rewards) == 0:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        # Convert to tensors and move to device
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        values = torch.tensor(values, device=self.device, dtype=torch.float32)
        
        advantages = torch.zeros_like(rewards)
        running_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            running_gae = delta + self.gamma * self.gae_lambda * running_gae
            advantages[t] = running_gae
        
        returns = advantages + values
        
        return advantages, returns

    def display_diagnostics(self):
        """Print diagnostic information"""
        if not self.training_stats['policy_loss']:
            return

        print("\n====== DIAGNOSTIC REPORT ======")

        # Recent averages (last 20 updates)
        recent_n = min(20, len(self.training_stats['policy_loss']))
        if recent_n > 0:
            recent_policy_loss = np.mean(self.training_stats['policy_loss'][-recent_n:])
            recent_value_loss = np.mean(self.training_stats['value_loss'][-recent_n:])
            recent_entropy = np.mean(self.training_stats['entropy'][-recent_n:])
            recent_clipfrac = np.mean(self.training_stats['clipfrac'][-recent_n:])
            
            print(f" . Policy Loss: {recent_policy_loss:.4f}")
            print(f" . Value Loss:  {recent_value_loss:.4f}")
            print(f" . Entropy:     {recent_entropy:.4f} {'⚠️ LOW' if recent_entropy < 0.5 else '✅'}")
            print(f" . Clip Frac:   {recent_clipfrac:.4f}")
            
            # Trend analysis
            if len(self.training_stats['entropy']) >= 40:
                older_entropy = np.mean(self.training_stats['entropy'][-40:-20])
                entropy_trend = recent_entropy - older_entropy
                print(f" . Entropy Trend: {'📈 INCREASING' if entropy_trend > 0.01 else '📉 DECREASING' if entropy_trend < -0.01 else '➡️ STABLE'}")
        
        print("====== END DIAGNOSTIC ======\n")

    def get_training_summary(self):
        """Get a summary of training statistics"""
        if not self.training_stats['policy_loss']:
            return "No training data available"
        
        # Ensure that stats are computed only if there's enough data
        avg_entropy = np.mean(self.training_stats['entropy'][-100:]) if len(self.training_stats['entropy']) >= 100 else (np.mean(self.training_stats['entropy']) if len(self.training_stats['entropy']) > 0 else 0)
        avg_policy_loss = np.mean(self.training_stats['policy_loss'][-100:]) if len(self.training_stats['policy_loss']) >= 100 else (np.mean(self.training_stats['policy_loss']) if len(self.training_stats['policy_loss']) > 0 else 0)

        return {
            'total_updates': self.update_count,
            'current_entropy': self.training_stats['entropy'][-1] if self.training_stats['entropy'] else 0,
            'avg_entropy': avg_entropy,
            'avg_policy_loss': avg_policy_loss
        }

    def reset_training_stats(self):
        """Reset training statistics (useful between sessions)"""
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clipfrac': []
        }
        self.update_count = 0 # Reset update count as well