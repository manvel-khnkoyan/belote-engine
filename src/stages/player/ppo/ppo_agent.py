import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOAgent:
    def __init__(self, network, lr=3e-4, gamma=0.97, gae_lambda=0.95, clip=0.2, 
                 entropy_coef=0.08, max_grad_norm=0.5, weight_decay=1e-4, 
                 value_loss_coef=0.5):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Chaotic environment optimized parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        # Training monitoring and diagnostics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clipfrac': []
        }
        self.last_stats = {}
        self.update_count = 0

    def act(self, type, probability, table, trump, valid_actions=None, training=False):
        probability = probability.to(self.device)
        table = table.to(self.device)
        trump = trump.to(self.device)

        with torch.no_grad():
            policy, value = self.network(type, probability, table, trump)
            policy = policy.squeeze()
            
            if valid_actions is not None:
                mask = torch.full_like(policy, float('-inf'))
                mask[valid_actions] = 0
                policy = policy + mask
            
            policy = F.softmax(policy, dim=-1)
            
            if torch.isnan(policy).any() or policy.sum() == 0:
                policy = torch.zeros_like(policy)
                if valid_actions is not None:
                    policy[valid_actions] = 1.0 / len(valid_actions)
                else:
                    policy.fill_(1.0 / len(policy))
            
            # Add exploration noise for chaotic environments
            if training:
                noise = torch.randn_like(policy) * 0.01
                policy = F.softmax(policy.log() + noise, dim=-1)
            
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return {
            'action_type': type,
            'action': action.item(),
            'value': value.squeeze().item(),
            'probability': probability.detach().squeeze().cpu(),
            'table': table.detach().squeeze().cpu(),
            'trump': trump.detach().squeeze().cpu(),
            'log_prob': log_prob.item(),
            'entropy': dist.entropy().item()
        }

    def learn(self, batch):
        """Simplified single-pass learning"""
        
        advantages, returns = self._compute_gae(batch['rewards'], batch['values'])
        
        # Robust normalization
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = len(batch['action_types'])
        actions_tensor = torch.tensor(batch['actions'], device=self.device, dtype=torch.long)
        old_log_probs = torch.tensor(batch['log_probs'], device=self.device, dtype=torch.float32)

        probs_batch = torch.stack([batch['probabilities'][i] for i in range(batch_size)]).to(self.device)
        table_batch = torch.stack([batch['tables'][i] for i in range(batch_size)]).to(self.device)
        trump_batch = torch.stack([batch['trumps'][i] for i in range(batch_size)]).to(self.device)
        action_type = batch['action_types'][0]
        
        # Forward pass
        new_policies, new_values = self.network(action_type, probs_batch, table_batch, trump_batch)
        new_policies = new_policies.squeeze(1) if new_policies.dim() > 2 else new_policies
        new_values = new_values.squeeze()
            
        dists = torch.distributions.Categorical(F.softmax(new_policies, dim=-1))
        new_log_probs = dists.log_prob(actions_tensor)
        entropy = dists.entropy().mean()
        
        # PPO loss
        ratios = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        entropy_loss = -self.entropy_coef * entropy
        
        loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        # Check gradients before update
        total_grad_norm_before = 0
        for param in self.network.parameters():
            if param.grad is not None:
                total_grad_norm_before += param.grad.data.norm(2).item() ** 2
        total_grad_norm_before = total_grad_norm_before ** 0.5
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        
        # Check gradients after backward
        total_grad_norm_after = 0
        for param in self.network.parameters():
            if param.grad is not None:
                total_grad_norm_after += param.grad.data.norm(2).item() ** 2
        total_grad_norm_after = total_grad_norm_after ** 0.5
        
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Calculate stats
        with torch.no_grad():
            clipfrac = ((ratios > (1 + self.clip)) | (ratios < (1 - self.clip))).float().mean()
        
        self.last_stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'clipfrac': clipfrac.item(),
            'grad_norm': total_grad_norm_after,
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
        }
        
        # Store stats for monitoring
        for key in ['policy_loss', 'value_loss', 'entropy', 'clipfrac']:
            self.training_stats[key].append(self.last_stats[key])

    
        return self.last_stats
    
    def _compute_gae(self, rewards, values):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE formula:
        δₜ = rₜ + γVₜ₊₁ - Vₜ
        Aₜ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...
        
        Args:
            rewards: List of rewards for each timestep
            values: List of value estimates for each timestep
        
        Returns:
            advantages: GAE advantages for each timestep
            returns: Value targets (advantages + values)
        """
        if len(rewards) == 0:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        # Convert to tensors
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        values = torch.tensor(values, device=self.device, dtype=torch.float32)
        
        # Initialize advantages tensor
        advantages = torch.zeros_like(rewards)
        
        # GAE calculation - work backwards through time
        running_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            # Calculate next value (0 for terminal state)
            if t == len(rewards) - 1:
                next_value = 0.0  # Terminal state has no next value
            else:
                next_value = values[t + 1]
            
            # Calculate TD error: δₜ = rₜ + γVₜ₊₁ - Vₜ
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # Update running GAE: Aₜ = δₜ + γλAₜ₊₁
            running_gae = delta + self.gamma * self.gae_lambda * running_gae
            
            # Store advantage for this timestep
            advantages[t] = running_gae
        
        # Calculate returns (value targets): Rₜ = Aₜ + Vₜ
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
        
        return {
            'total_updates': self.update_count,
            'current_entropy': self.training_stats['entropy'][-1] if self.training_stats['entropy'] else 0,
            'avg_entropy': np.mean(self.training_stats['entropy'][-100:]) if len(self.training_stats['entropy']) >= 100 else np.mean(self.training_stats['entropy']),
            'avg_policy_loss': np.mean(self.training_stats['policy_loss'][-100:]) if len(self.training_stats['policy_loss']) >= 100 else np.mean(self.training_stats['policy_loss'])
        }

    def reset_training_stats(self):
        """Reset training statistics (useful between sessions)"""
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clipfrac': []
        }