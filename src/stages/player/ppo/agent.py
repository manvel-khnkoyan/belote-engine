import torch
import torch.optim as optim
import torch.nn.functional as F

class PPOAgent:

    def __init__(self, network, lr=3e-4, gamma=0.99, clip=0.2, entropy_coef=0.01, max_grad_norm=0.5, weight_decay=1e-4):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.gamma = gamma
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

    def act(self, type, probability, table, trump, valid_actions=None):
        probability = probability.to(self.device)
        table = table.to(self.device)
        trump = trump.to(self.device)

        with torch.no_grad():
            # Get policy and value from network (note: using Action.TYPE_CARD_PLAY = 1)
            policy, value = self.network(type, probability, table, trump)
            policy = policy.squeeze()
            
            # Mask invalid actions if provided
            if valid_actions is not None:
                # Create mask for valid actions only
                mask = torch.full_like(policy, float('-inf'))
                mask[valid_actions] = 0
                policy = policy + mask
            
            # Convert to probabilities and sample
            policy = F.softmax(policy, dim=-1)
            
            # Prevent NaN by ensuring we have valid probabilities
            if torch.isnan(policy).any() or policy.sum() == 0:
                # Fallback: uniform over valid actions
                policy = torch.zeros_like(policy)
                if valid_actions is not None:
                    policy[valid_actions] = 1.0 / len(valid_actions)
                else:
                    policy.fill_(1.0 / len(policy))
            
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
            'log_prob': log_prob.item()
        }

    def learn(self, batch):
        # Use dictionary access instead of attribute access
        returns = self._compute_returns(batch['rewards'])
        values_tensor = torch.tensor(batch['values'], device=self.device, dtype=torch.float32)
        advantages = returns - values_tensor

        # Normalize advantages if we have variance
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert batch actions and log_probs to tensors
        actions_tensor = torch.tensor(batch['actions'], device=self.device, dtype=torch.long)
        old_log_probs = torch.tensor(batch['log_probs'], device=self.device, dtype=torch.float32)

        # Stack all tensors first to create proper batch dimensions
        batch_size = len(batch['action_types'])
        
        # Stack probabilities, tables, and trumps into batched tensors
        probabilities_batch = torch.stack([batch['probabilities'][i] for i in range(batch_size)]).to(self.device)
        tables_batch = torch.stack([batch['tables'][i] for i in range(batch_size)]).to(self.device)
        trumps_batch = torch.stack([batch['trumps'][i] for i in range(batch_size)]).to(self.device)
        
        # Get action types (assuming they're all the same for this batch)
        action_type = batch['action_types'][0]  # Assuming homogeneous batch
        
        # Get new policies and values from the network for the batched states
        new_policies, new_values = self.network(action_type, probabilities_batch, tables_batch, trumps_batch)
        
        # Squeeze to remove unnecessary dimensions
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
        
        loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        # Update with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
    
    def _compute_returns(self, rewards):
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, device=self.device, dtype=torch.float32)