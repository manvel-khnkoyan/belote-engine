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
from src.stages.player.memory import Memory
from src.stages.player.actions import Action, ActionCardMove

class PPOBeloteAgent:
    def __init__(self, network, lr=0.0003, gamma=0.99, gae_lambda=0.95, 
                 policy_clip=0.2, n_epochs=10, memorize=False):

        self.name = 'AI'
        self.memorize = memorize
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = Memory()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

        # Propability is personalized for each agent
        self.probability = None 
        # Current agent index in the environment
        self.env_index = 0

    def _convert_env_index(self, env_player):
        return (self.env_index + env_player) % 4

    def init(self, env, probability = None, env_index = 0):
        # Each agent has its own probability memory
        self.probability = Probability() if probability is None else probability
        self.env_index = env_index

        for card in env.deck.hands[env_index]: # My hand
            self.probability.update(0, card.suit, card.rank, 1)

    def observe(self, env_player: int, action: Action):
        player = self._convert_env_index(env_player)

        # Update card knowledge for the player who played the card
        if isinstance(action, ActionCardMove):
            self.probability.update(player, action.card.suit, action.card.rank, -1)

    def choose_action(self, env):
        self.network.eval()

        # Get the canonical transformation for the suits
        transform, _ = suits_canonical_transformer(self.probability)
        
        # Prepare input tensors for network
        probs_tensor = self.probability.copy().change_suits(transform).to_tensor().to(self.device)
        probs_tensor = probs_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4, 8]
        
        # For table tensor - make sure change_suits returns a new object and doesn't modify in place
        transformed_table = env.table.copy().change_suits(transform)
        table_tensor = transformed_table.to_tensor().to(self.device)
        table_tensor = table_tensor.unsqueeze(0)  # [1, 3, 8, 4]
        
        # For trump tensor - same approach
        transformed_trump = env.trump.copy().change_suits(transform)
        trump_tensor = transformed_trump.to_tensor().to(self.device)
        trump_tensor = trump_tensor.unsqueeze(0)  # [1, 4]

        network_actions = []
        
       # Forward pass to get policy and value based on action type
        with torch.no_grad():
            # action type: PLAY
            valid_cards = env.valid_cards()  # This is fine if it just returns references without modifying
            if len(valid_cards) > 0:
                # Create new Card objects for transformation to avoid modifying originals
                transformed_cards = [card.copy().change_suit(transform) for card in valid_cards]
                
                # Get policy and value for card action type
                card_policy, card_value = self.network(Action.TYPE_PLAY, probs_tensor, table_tensor, trump_tensor)
                
                # Choose card based on policy
                chosen_card, card_masks, action_item, log_prob = self._choose_card(transformed_cards, card_policy)
                
                # Map back to original card
                original_card = valid_cards[transformed_cards.index(chosen_card)]
                
                network_actions.append({
                    'type': Action.TYPE_PLAY,
                    'move': original_card,
                    'item': action_item,
                    'masks': card_masks,
                    'value': card_value,
                    'log_prob': log_prob
                })

            # action type: BELOTE
            # Add other action types here when implemented
            # ...

        # If no actions are available, raise an exception
        if not network_actions:
            raise ValueError("No valid actions available")

        # Best action based on the highest value
        best_network_action = max(network_actions, key=lambda x: x['value'].item())

        # If memorize is True, store the action in memory
        if self.memorize:
            self.memory.store(
                probability_tensor=probs_tensor.cpu().detach().clone(),
                table_tensor=table_tensor.cpu().detach().clone(),
                trump_tensor=trump_tensor.cpu().detach().clone(),
                action_type=best_network_action['type'],
                action=best_network_action['item'],
                actions_mask=best_network_action['masks'].cpu().numpy(),
                value=best_network_action['value'].item(),
                reward=0,  # Reward will be set later when the trick ends
                log_prob=best_network_action['log_prob'].item()
            )
        
        # Return action object
        if best_network_action['type'] == Action.TYPE_PLAY:
            return ActionCardMove(best_network_action['move'])

        raise NotImplemented(f"{best_network_action['type']} action type not implemeted")

    def _choose_card(self, valid_cards, card_policy):
        # Create action mask (1 for valid actions, 0 for invalid)
        action_mask = torch.zeros(self.network.total_actions, dtype=torch.float32).to(self.device)
        
        # Map each valid card to an action index
        valid_card_map = {}  # Map to store action_idx -> original card
        for card in valid_cards:
            # Convert card to action index (rank * num_suits + suit)
            action_idx = card.rank * self.network.num_suits + card.suit
            action_mask[action_idx] = 1.0
            valid_card_map[action_idx] = card
                
        # Apply mask to policy
        masked_policy = self._safe_normalize(card_policy.squeeze(0), action_mask)
        
        # Sample action from the masked policy
        action_dist = torch.distributions.Categorical(masked_policy)
        action_idx = action_dist.sample()
        action_item = action_idx.item()
        log_prob = action_dist.log_prob(action_idx)
        
        chosen_card = valid_card_map[action_item]

        return chosen_card, masked_policy, action_item, log_prob

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
        old_action_types = torch.tensor(self.memory.action_types, dtype=torch.int64).to(self.device)
        
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
                    
                batch_trump_tensor = torch.stack([self.memory.trump_tensors[i].to(self.device) for i in batch_indices])
                # Fix extra dimension in trump tensor if needed
                if batch_trump_tensor.dim() > 2:  # If it has more than 2 dimensions
                    batch_trump_tensor = batch_trump_tensor.squeeze(1)  # Remove the extra dimension
                    
                batch_actions = old_actions[batch_indices]
                batch_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_action_masks = old_action_masks[batch_indices]
                batch_action_types = old_action_types[batch_indices]
                
                # Initialize losses
                actor_losses = []
                critic_losses = []
                entropy_losses = []
                
                # Process each action type separately
                # Get unique action types in the batch
                unique_action_types = torch.unique(batch_action_types, dim=0)
                
                for action_type in unique_action_types:
                    # Find indices in the batch that have this action type
                    type_mask = batch_action_types == action_type
                    type_indices = torch.nonzero(type_mask).squeeze(1)
                    
                    if len(type_indices) == 0:
                        continue
                    
                    # Get data for this action type
                    type_probs = batch_probs_tensors[type_indices]
                    type_tables = batch_tables[type_indices]
                    type_trump = batch_trump_tensor[type_indices]
                    type_actions = batch_actions[type_indices]
                    type_log_probs = batch_log_probs[type_indices]
                    type_returns = batch_returns[type_indices]
                    type_advantages = batch_advantages[type_indices]
                    type_action_masks = batch_action_masks[type_indices]
                    
                    # Forward pass with the appropriate head based on action type
                    policy, values_new = self.network(
                        action_type,  # Pass the action type directly to the network
                        type_probs,
                        type_tables,
                        type_trump,
                    )
                    
                    # Apply action masks
                    masked_policy = self._safe_normalize(policy, type_action_masks)
                    
                    # Get log probabilities for actions
                    action_dists = torch.distributions.Categorical(masked_policy)
                    new_log_probs = action_dists.log_prob(type_actions)
                    
                    # Calculate ratio (π_new / π_old)
                    ratios = torch.exp(new_log_probs - type_log_probs)
                    
                    # Calculate surrogate losses
                    surr1 = ratios * type_advantages
                    surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * type_advantages
                    
                    # Actor loss for this action type
                    type_actor_loss = -torch.min(surr1, surr2).mean()
                    actor_losses.append(type_actor_loss)
                    
                    # Critic loss for this action type
                    type_critic_loss = nn.MSELoss()(values_new.squeeze(), type_returns)
                    critic_losses.append(type_critic_loss)
                    
                    # Entropy for this action type
                    type_entropy = action_dists.entropy().mean()
                    entropy_losses.append(type_entropy)
                
                # Combine losses from all action types
                if actor_losses:
                    actor_loss = sum(actor_losses) / len(actor_losses)
                    critic_loss = sum(critic_losses) / len(critic_losses)
                entropy = sum(entropy_losses) / len(entropy_losses)
                
                # Total loss
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