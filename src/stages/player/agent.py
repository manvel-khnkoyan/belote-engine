"""
Each Agent plays as player N: 0
It means each agent has its own memory
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from src.card import Card
from src.canonical.suits_transformer import suits_canonical_transformer
from src.states.probability import Probability
class Memory:
    def __init__(self):
        self.probability_tensors = []   # Store probability tensors
        self.tables_tensors = []        # Store table tensors
        self.trump_tensors = []         # Store trump tensors
        self.action_types = []
        self.actions = []
        self.actions_masks = []
        self.values = []
        self.rewards = []
        self.log_probs = []

        self.seed = int(time.time()) % 1000

    def __len__(self):
        return len(self.rewards)
        
    def store(self, probability_tensor, table_tensor, trump_tensor, action_type, action, actions_mask, value, reward, log_prob=None):
        """
        Store a transition in memory using tensor representations
        
        Args:
            probability_tensor: Tensor representation of probability matrix
            table_tensor: Tensor representation of table cards
            trump_tensor: Tensor representation of trump
            action_type: Type of action taken
            action: The action taken
            actions_mask: Mask of valid actions
            value: Value estimate from network
            reward: Reward received
            log_prob: Log probability of action
        """
        self.probability_tensors.append(probability_tensor)
        self.tables_tensors.append(table_tensor)
        self.trump_tensors.append(trump_tensor)
        self.action_types.append(action_type)
        self.actions.append(action)
        self.actions_masks.append(actions_mask)
        self.values.append(value)
        self.rewards.append(reward)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        
    def clear(self):
        self.probability_tensors = []
        self.tables_tensors = []
        self.trump_tensors = []
        self.action_types = []
        self.actions = []
        self.actions_masks = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        
    def random_batches(self, batch_size):
        batch_start = np.arange(0, len(self.rewards), batch_size)
        indices = np.arange(len(self.rewards), dtype=np.int64)

        self.seed += 1 
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        
        batches = [indices[i:i+batch_size] for i in batch_start]

        return batches
    
    def update_last_reward(self, reward):
        if len(self.rewards) > 0:
            self.rewards[-1] = reward

class PPOBeloteAgent:
    def __init__(self, network, lr=0.0003, gamma=0.99, gae_lambda=0.95, 
                 policy_clip=0.2, n_epochs=10):

        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = Memory()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

    def reset_memory(self, env, player=0):
        # Each agent has its own probability memory
        # Separate from the environment
        self.probability = Probability()

        for card in env.deck.hands[player]:
            self.probability.update(player, card.suit, card.rank, 1)

    def update_memory(self, player, card):
        # Update card knowledge for the player who played the card
        self.probability.update(player, card.suit, card.rank, -1)
    
    def choose_action(self, env, memorize=False) -> Card:
        self.network.eval()

        # Get the canonical transformation for the suits
        transform, _ = suits_canonical_transformer(self.probability)
        
        # Get Env variables as tensors with proper shapes for the network
        probs_tensor = self.probability.copy().change_suits(transform).to_tensor().to(self.device)
        # Add batch and channel dimensions for Conv3D (needs to be [batch, channel, players, suits, ranks])
        probs_tensor = probs_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4, 8]
        
        table_tensor = env.table.copy().change_suits(transform).to_tensor().to(self.device)
        # Add batch dimension (needs to be [batch, cards, ranks, suits])
        table_tensor = table_tensor.unsqueeze(0)  # [1, 3, 8, 4]
        
        trump_tensor = env.trump.copy().change_suits(transform).to_tensor().to(self.device)
        # Add batch dimension (needs to be [batch, suits])
        trump_tensor = trump_tensor.unsqueeze(0)  # [1, 4]
        
        # Get valid cards from environment
        valid_cards = env.valid_cards()
        
        # Create action mask (1 for valid actions, 0 for invalid)
        action_mask = torch.zeros(self.network.total_actions, dtype=torch.float32).to(self.device)
        
        # Map each valid card to an action index
        valid_card_map = {}  # Map to store action_idx -> original card
        for card in valid_cards:
            # Apply canonical transformation to the by creating a new Card object
            canonical_card = Card(card.suit, card.rank).change_suit(transform) 
            # Convert card to action index (rank * num_suits + suit)
            action_idx = canonical_card.rank * self.network.num_suits + canonical_card.suit
            action_mask[action_idx] = 1.0
            valid_card_map[action_idx] = card  # Store mapping from action index to original card
        
        # Forward pass to get policy and value
        with torch.no_grad():
            policy, value = self.network(probs_tensor, table_tensor, trump_tensor)
            
            # Apply mask to policy
            # We ensure only valid actions have non-zero probability
            masked_policy = self._safe_normalize(policy.squeeze(0), action_mask)
            
            # Sample action from the masked policy
            action_dist = torch.distributions.Categorical(masked_policy)
            action_idx = action_dist.sample()
            action_item = action_idx.item()
            log_prob = action_dist.log_prob(action_idx)
            chosen_card = valid_card_map[action_item]

        # If memorize is True, store the action in memory
        if memorize:
            # Make CPU copies of tensors and detach from computation graph for storage
            self.memory.store(
                probability_tensor=probs_tensor.cpu().detach().clone(),
                table_tensor=table_tensor.cpu().detach().clone(),
                trump_tensor=trump_tensor.cpu().detach().clone(),
                action_type=1,  # Assuming action type 1 for card play
                action=action_item,
                actions_mask=action_mask.cpu().numpy(),
                value=value.item(),
                reward=0,  # Reward will be set later when the trick ends
                log_prob=log_prob.item()
            )
        
        return chosen_card

    def learn(self, batch_size=64):
        if len(self.memory.rewards) == 0 or len(self.memory.rewards) < batch_size:
            return  # Nothing to learn from
            
        self.network.train()
        
        # Convert rewards to tensor
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.memory.values, dtype=torch.float32).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(rewards, values)
        
        # Get batches
        batches = self.memory.random_batches(batch_size)
        
        # Convert memory data to tensors if they aren't already
        old_actions = torch.tensor(self.memory.actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32).to(self.device)
        old_action_masks = torch.tensor(np.array(self.memory.actions_masks), dtype=torch.float32).to(self.device)
        
        # Learning over multiple epochs
        for _ in range(self.n_epochs):
            for batch in batches:
                # Get batch indices
                batch_indices = torch.tensor(batch, dtype=torch.int64)
                
                # Prepare batch data
                batch_probs_tensors = torch.stack([self.memory.probability_tensors[i].to(self.device) for i in batch_indices])
                # Fix extra dimension in probability tensor
                if batch_probs_tensors.dim() > 5:  # If it has more than 5 dimensions
                    batch_probs_tensors = batch_probs_tensors.squeeze(2)  # Remove the extra dimension
                    
                batch_tables = torch.stack([self.memory.tables_tensors[i].to(self.device) for i in batch_indices])
                # Fix extra dimension in table tensor
                if batch_tables.dim() > 4:  # If it has more than 4 dimensions
                    batch_tables = batch_tables.squeeze(1)  # Remove the extra dimension
                    
                trump_tensor = torch.stack([self.memory.trump_tensors[i].to(self.device) for i in batch_indices])
                # Fix extra dimension in trump tensor if needed
                if trump_tensor.dim() > 2:  # If it has more than 2 dimensions
                    trump_tensor = trump_tensor.squeeze(1)  # Remove the extra dimension
                    
                batch_actions = old_actions[batch_indices]
                batch_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_action_masks = old_action_masks[batch_indices]
                
                # Forward pass with corrected tensor shapes
                policy, values_new = self.network(
                    batch_probs_tensors,
                    batch_tables,
                    trump_tensor,
                )
                
                # Apply action masks
                masked_policy = self._safe_normalize(policy, batch_action_masks)
                
                # Get log probabilities for actions
                action_dists = torch.distributions.Categorical(masked_policy)
                new_log_probs = action_dists.log_prob(batch_actions)
                
                # Calculate ratio (π_new / π_old)
                ratios = torch.exp(new_log_probs - batch_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * batch_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(values_new.squeeze(), batch_returns)
                
                # Total loss
                entropy = action_dists.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def _safe_normalize(self, policy, mask, epsilon=1e-8):
        # Apply mask
        masked = policy * mask
        
        # Check if we're dealing with a batch or single policy
        if masked.dim() > 1:
            # Batch case - handle each row
            result = masked.clone()
            for i in range(masked.shape[0]):
                row_sum = masked[i].sum()
                if row_sum > epsilon:
                    result[i] = masked[i] / row_sum
                else:
                    # If sum is too small, create uniform distribution ONLY over valid actions
                    valid_mask = mask[i] > 0
                    result[i] = valid_mask.float() / valid_mask.sum()
            
            return result

        # Single policy case
        sum_masked = masked.sum()
        if sum_masked > epsilon:
            return masked / sum_masked
        
        # If sum is too small, create uniform distribution ONLY over valid actions
        valid_mask = mask > 0
        return valid_mask.float() / valid_mask.sum()

    def _compute_advantages(self, rewards, values):
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # We need values for t+1 for GAE calculation, so add a 0 at the end
        next_value = 0
        
        # Going backwards to calculate returns and advantages
        gae = 0
        for t in reversed(range(len(rewards))):
            # For the last step, there's no next value
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            # Calculate TD error: r_t + γ V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_val - values[t]
            
            # Calculate GAE
            gae = delta + self.gamma * self.gae_lambda * gae
            
            # Store advantage and return
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        if len(advantages) > 1:  # Only normalize if we have more than one advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def save(self, path):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])