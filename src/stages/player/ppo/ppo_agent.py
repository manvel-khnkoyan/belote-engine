import torch
import torch.optim as optim
import torch.nn.functional as F
# ================================
# OPTIMIZED agent.py (PPOAgent)
# ================================

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOAgent:
    def __init__(self, network, lr=3e-4, gamma=0.9, gae_lambda=0.8, clip=0.2, 
                 entropy_coef=0.02, max_grad_norm=0.5, weight_decay=1e-4, 
                 ppo_epochs=3, value_loss_coef=0.5, kl_threshold=0.01):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Chaotic environment optimized parameters
        self.gamma = gamma  # Lower for short-term focus
        self.gae_lambda = gae_lambda  # Lower for less bootstrapping
        self.clip = clip
        self.entropy_coef = entropy_coef  # Higher for exploration
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs  # Multiple epochs
        self.value_loss_coef = value_loss_coef
        self.kl_threshold = kl_threshold
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        # Training monitoring and diagnostics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clipfrac': [],
            'early_stops': 0
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

    def _update_epoch(self, batch, old_policies=None):
        """Single epoch update with proper KL calculation"""
        advantages, returns = self._compute_gae(batch['rewards'], batch['values'])
        
        # Debug: Check advantage statistics
        print(f"    Advantages - Mean: {advantages.mean():.6f}, Std: {advantages.std():.6f}")
        print(f"    Returns - Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
        
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
        
        # Debug: Check ratio statistics
        print(f"    Ratios - Min: {ratios.min():.4f}, Max: {ratios.max():.4f}, Mean: {ratios.mean():.4f}")
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        entropy_loss = -self.entropy_coef * entropy
        
        loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        # Debug: Check gradients before update
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
        
        print(f"    Grad Norm: {total_grad_norm_after:.6f}")
        
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # FIXED: Proper KL calculation using old policies stored before epochs
        with torch.no_grad():
            if old_policies is not None:
                old_policy_dist = torch.distributions.Categorical(F.softmax(old_policies, dim=-1))
                new_policy_dist = torch.distributions.Categorical(F.softmax(new_policies.detach(), dim=-1))
                kl_div = torch.distributions.kl.kl_divergence(old_policy_dist, new_policy_dist).mean()
            else:
                kl_div = torch.tensor(0.0)
            
            clipfrac = ((ratios > (1 + self.clip)) | (ratios < (1 - self.clip))).float().mean()
        
        self.last_stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_div.item(),
            'clipfrac': clipfrac.item(),
            'grad_norm': total_grad_norm_after,
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
        }
        
        return self.last_stats

    def learn(self, batch):
        """Enhanced learning with multiple epochs and early stopping"""
        self.update_count += 1
        
        # Store old policy ONCE before all epochs
        with torch.no_grad():
            batch_size = len(batch['action_types'])
            probs_batch = torch.stack([batch['probabilities'][i] for i in range(batch_size)]).to(self.device)
            table_batch = torch.stack([batch['tables'][i] for i in range(batch_size)]).to(self.device)
            trump_batch = torch.stack([batch['trumps'][i] for i in range(batch_size)]).to(self.device)
            action_type = batch['action_types'][0]
            
            old_policies, _ = self.network(action_type, probs_batch, table_batch, trump_batch)
            old_policies = old_policies.squeeze(1) if old_policies.dim() > 2 else old_policies
        
        print(f"\n--- Learning Update {self.update_count} ---")
        print(f"Batch size: {batch_size}")
        print(f"Unique rewards in batch: {len(set(batch['rewards']))}")
        print(f"Reward range: [{min(batch['rewards']):.4f}, {max(batch['rewards']):.4f}]")
        
        # Multiple epochs with enhanced early stopping
        epochs_completed = 0
        for epoch in range(self.ppo_epochs):
            print(f"  Epoch {epoch + 1}/{self.ppo_epochs}:")
            stats = self._update_epoch(batch, old_policies)
            epochs_completed = epoch + 1
            
            # Store stats for monitoring
            for key, value in stats.items():
                if key in self.training_stats:
                    self.training_stats[key].append(value)
            
            # Enhanced early stopping with diagnostic information
            if len(self.training_stats['kl_divergence']) > 0:
                recent_kl = np.mean(self.training_stats['kl_divergence'][-3:])  # Last 3 updates
                if recent_kl > self.kl_threshold:
                    self.training_stats['early_stops'] += 1
                    print(f"    [EARLY STOP] KL too high: {recent_kl:.6f}")
                    break
        
        # Diagnostic information every 10 updates (more frequent)
        if self.update_count % 10 == 0:
            self.print_diagnostics()
        
        # Store summary stats
        self.last_stats = {
            'epochs_completed': epochs_completed,
            'recent_kl': np.mean(self.training_stats['kl_divergence'][-5:]) if self.training_stats['kl_divergence'] else 0,
            'recent_entropy': np.mean(self.training_stats['entropy'][-5:]) if self.training_stats['entropy'] else 0,
            'early_stops_total': self.training_stats['early_stops']
        }
    
        return self.last_stats
    
    def _compute_gae(self, rewards, values):
        """Enhanced GAE for chaotic environments"""
        if len(rewards) == 0:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        values = torch.tensor(values, device=self.device, dtype=torch.float32)
        
        next_values = torch.cat([values[1:], torch.zeros(1, device=self.device)])
        deltas = rewards + self.gamma * next_values - values
        
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            gae = deltas[i] + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = advantages + values
        
        return advantages, returns

    def print_diagnostics(self):
        """Print comprehensive diagnostic information"""
        if not self.training_stats['policy_loss']:
            return
        
        print(f"\n=== DIAGNOSTIC REPORT (Update {self.update_count}) ===")
        
        # Recent averages (last 20 updates)
        recent_n = min(20, len(self.training_stats['policy_loss']))
        if recent_n > 0:
            recent_policy_loss = np.mean(self.training_stats['policy_loss'][-recent_n:])
            recent_value_loss = np.mean(self.training_stats['value_loss'][-recent_n:])
            recent_entropy = np.mean(self.training_stats['entropy'][-recent_n:])
            recent_kl = np.mean(self.training_stats['kl_divergence'][-recent_n:])
            recent_clipfrac = np.mean(self.training_stats['clipfrac'][-recent_n:])
            
            print(f"Recent Averages (last {recent_n} updates):")
            print(f"  Policy Loss: {recent_policy_loss:.4f}")
            print(f"  Value Loss:  {recent_value_loss:.4f}")
            print(f"  Entropy:     {recent_entropy:.4f} {'⚠️ LOW' if recent_entropy < 0.5 else '✅'}")
            print(f"  KL Div:      {recent_kl:.4f} {'⚠️ HIGH' if recent_kl > self.kl_threshold else '✅'}")
            print(f"  Clip Frac:   {recent_clipfrac:.4f}")
            print(f"  Early Stops: {self.training_stats['early_stops']}")
            
            # Trend analysis
            if len(self.training_stats['entropy']) >= 40:
                older_entropy = np.mean(self.training_stats['entropy'][-40:-20])
                entropy_trend = recent_entropy - older_entropy
                print(f"  Entropy Trend: {'📈 INCREASING' if entropy_trend > 0.01 else '📉 DECREASING' if entropy_trend < -0.01 else '➡️ STABLE'}")
        
        print("=== END DIAGNOSTIC ===\n")

    def get_training_summary(self):
        """Get a summary of training statistics"""
        if not self.training_stats['policy_loss']:
            return "No training data available"
        
        return {
            'total_updates': self.update_count,
            'total_early_stops': self.training_stats['early_stops'],
            'current_entropy': self.training_stats['entropy'][-1] if self.training_stats['entropy'] else 0,
            'current_kl': self.training_stats['kl_divergence'][-1] if self.training_stats['kl_divergence'] else 0,
            'avg_entropy': np.mean(self.training_stats['entropy'][-100:]) if len(self.training_stats['entropy']) >= 100 else np.mean(self.training_stats['entropy']),
            'avg_kl': np.mean(self.training_stats['kl_divergence'][-100:]) if len(self.training_stats['kl_divergence']) >= 100 else np.mean(self.training_stats['kl_divergence'])
        }

    def reset_training_stats(self):
        """Reset training statistics (useful between sessions)"""
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clipfrac': [],
            'early_stops': 0
        }
