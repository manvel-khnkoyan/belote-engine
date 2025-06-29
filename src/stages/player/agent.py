import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
import os
from src.stages.player.actions import Action
from src.states.probability import Probability
from src.stages.player.actions import ActionCardMove
from src.canonical.suits_canonical_transformer import SuitsCanonicalTransformer
from src.stages.player.actions import Action, ActionCardMove
from src.deck import Deck

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
            # Get policy and value from network (note: using Action.TYPE_PLAY = 1)
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

        return action.item(), \
            value.item(), \
            probability.detach().cpu(), \
            table.detach().cpu(), \
            trump.detach().cpu(), \
            log_prob.item()

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

        # Get new policies and values from the network for the current states
        new_policies = []
        new_values = []
        for i in range(len(batch['action_types'])):
            action_type = batch['action_types'][i]
            probs_tensor = batch['probabilities'][i].to(self.device).unsqueeze(0)  # Add batch dimension
            table_tensor = batch['tables'][i].to(self.device).unsqueeze(0)  # Add batch dimension
            trump_tensor = batch['trumps'][i].to(self.device).unsqueeze(0)  # Add batch dimension

            # Get policy and value from the network
            policy, value = self.network(action_type, probs_tensor, table_tensor, trump_tensor)
            new_policies.append(policy.squeeze())
            new_values.append(value.squeeze())
        
        # Stack tensors properly
        new_policies = torch.stack(new_policies)
        new_values = torch.stack(new_values)
            
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


class PPOBeloteAgent:
    
    def __init__(self, network, training=False):
        self.training = training
        self.seed = int(time.time() * 1000) % (2**32 - 1)
        self.reset_batch()
        self.ppo = PPOAgent(network)

    def reset_batch(self):
        self.action_types = []
        self.actions = []
        self.actions_masks = []
        self.probabilities = []
        self.tables = []
        self.trumps = []
        self.values = []
        self.log_probs = []
        self.rewards = []

    def init(self, env, env_index=0, probability=None):
        # Store the environment index
        self.env_index = env_index
        
        # Initialize Probability
        self.init_probability(env, probability)

    
    def init_probability(self, env, probability=None):
        # Initialize the probability matrix for the agent    
        self.probability = probability if probability else Probability()

        # Reset My Hands Probability
        for card in env.deck.hands[self.env_index]:
            self.probability.update(0, card.suit, card.rank, 1)


    def updated_rewards(self, reward_value, last_n=1, decay_factor=2):
        for i in range(last_n):
            idx = len(self.rewards) - 1 - i
            if idx >= 0:
                self.rewards[idx] += reward_value / (decay_factor ** i)


    def observe(self, player, action):
        # Every Agent observation is unique for themselves
        if isinstance(action, ActionCardMove):
            self.probability.update(player, action.card.suit, action.card.rank, -1)

    
    def choose_action(self, env):
        # Get state representation
        transformer = SuitsCanonicalTransformer(self.probability) if self.probability else None
        transform = transformer.get_transform_function() if transformer else None
        
        # Get tensors (with error handling)
        probability_tensor = (self.probability.copy().change_suits(transform).to_tensor().unsqueeze(0))
        table_tensor = (env.table.copy().change_suits(transform).to_tensor().unsqueeze(0))
        trump_tensor = (env.trump.copy().change_suits(transform).to_tensor().unsqueeze(0))
        
        # Get valid actions
        valid_cards = env.valid_cards()
        valid_indices = []
        card_map = {}
        
        for card in valid_cards:
            if transform is not None:
                transformed = card.copy().change_suits(transform)
                idx = transformed.rank * 4 + transformed.suit
            else:
                idx = card.rank * 4 + card.suit
                
            valid_indices.append(idx)
            card_map[idx] = card
        
        # In tensorize valid indices
        action_idx, value, probability, table, trump, log_prob = self.ppo.act(
            Action.TYPE_PLAY,
            probability_tensor, 
            table_tensor, 
            trump_tensor, 
            valid_indices
        )

        if self.training:
            # Store the action in the batch - keep the tensors without batch dimension for storage efficiency
            self.action_types.append(Action.TYPE_PLAY)
            self.actions.append(action_idx)
            self.probabilities.append(probability.squeeze().cpu())
            self.tables.append(table.squeeze().cpu())
            self.trumps.append(trump.squeeze().cpu())
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.rewards.append(0)
        
        # Check if action_idx is valid
        if action_idx in valid_indices:
            return ActionCardMove(card_map[action_idx])
        
        raise ValueError(f"Invalid action index: {action_idx}. Valid indices: {valid_indices}")
    
    def learn(self, batch_size=64):
        if len(self.actions) == 0:
            return

        self.seed += 1
        rng = np.random.default_rng(self.seed)
        
        # Handle case where batch_size > available actions
        actual_batch_size = min(batch_size, len(self.actions))
        random_indices = rng.choice(len(self.actions), size=actual_batch_size, replace=False)

        self.ppo.learn({
            'action_types': [self.action_types[i] for i in random_indices],
            'actions': [self.actions[i] for i in random_indices],
            'probabilities': [self.probabilities[i] for i in random_indices],
            'tables': [self.tables[i] for i in random_indices],
            'trumps': [self.trumps[i] for i in random_indices],
            'values': [self.values[i] for i in random_indices],
            'log_probs': [self.log_probs[i] for i in random_indices],
            'rewards': [self.rewards[i] for i in random_indices]
        })

    def save(self, path):
        torch.save({
            'network_state_dict': self.ppo.network.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
        }, path)

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.ppo.device)
            self.ppo.network.load_state_dict(checkpoint['network_state_dict'])
            self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return True
        return False