import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import namedtuple

from src.phases.play.core.agent import Agent
from src.phases.play.core.state import State
from src.phases.play.core.actions import Action, ActionPlayCard
from src.phases.play.core.record import Record
from src.phases.play.ppo.network import PPONetwork

# Simple container for batched state tensors
BatchedState = namedtuple('BatchedState', ['probabilities', 'tables', 'history', 'legal_actions'])


class PpoAgent(Agent):
    def __init__(self, network: PPONetwork, rng: np.random.Generator | None = None):
        super().__init__()
        self.network = network
        self.device = next(network.parameters()).device
        self.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Dict[str, Any] | None]:
        """
        Choose an action based on the current game state.
        """
        # Filter for PlayCard actions
        play_card_actions = [a for a in actions if isinstance(a, ActionPlayCard)]
        
        if not play_card_actions:
            # No card play actions available, return first action
            return actions[0], None

        # Create mask for valid actions
        valid_indices = []
        action_map = {}
        
        for action in play_card_actions:
            idx = action.card.suit * 8 + action.card.rank
            valid_indices.append(idx)
            action_map[idx] = action
            
        mask = torch.full((32,), -float('inf'), device=self.device)
        for idx in valid_indices:
            mask[idx] = 0

        # Prepare state for network
        state_batch = self._batch_state(state)
        state_batch.legal_actions = mask.unsqueeze(0)

        # Forward Pass
        with torch.no_grad():
            outputs = self.network(state_batch)
        
        # Get card policy logits
        card_logits = outputs['card_policy'][0]  # [32]
            
        masked_logits = card_logits # Already masked in network
        probs = torch.softmax(masked_logits, dim=0)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx_tensor = dist.sample()
        action_idx = action_idx_tensor.item()
        
        chosen_action = action_map[action_idx]
        
        # Prepare logs for training
        log_prob = dist.log_prob(action_idx_tensor)
        value = outputs['value'][0]
        
        log = {
            'log_prob': log_prob.item(),
            'value': value.item(),
            'state': {
                'probabilities': state_batch.probabilities.cpu().numpy() if isinstance(state_batch.probabilities, torch.Tensor) else state_batch.probabilities,
                'tables': state_batch.tables.cpu().numpy() if isinstance(state_batch.tables, torch.Tensor) else state_batch.tables,
                'history': state_batch.history.cpu().numpy() if isinstance(state_batch.history, torch.Tensor) else state_batch.history,
            },
            'action_idx': action_idx,
            'mask': mask.cpu().numpy()
        }
        
        return chosen_action, log

    def learn(self, records: List[Record]):
        """
        Train the agent based on the collected records.
        Assumes record.log contains 'advantage' and 'return' keys.
        """
        if not records:
            return {}

        # Hyperparameters
        clip_param = 0.2
        value_loss_coef = 0.5
        entropy_coef = 0.01
        max_grad_norm = 0.5
        ppo_epochs = 4
        batch_size = 64  # Mini-batch size
        
        # Collate data into tensors
        probs = torch.tensor(
            np.array([r.log['state']['probabilities'] for r in records]), 
            dtype=torch.float32, 
            device=self.device
        ).squeeze(1)
        tables = torch.tensor(
            np.array([r.log['state']['tables'] for r in records]), 
            dtype=torch.long, 
            device=self.device
        ).squeeze(1)
        histories = torch.tensor(
            np.array([r.log['state']['history'] for r in records]), 
            dtype=torch.float32, 
            device=self.device
        ).squeeze(1)
        
        # Actions & Old Log Probs
        actions = torch.tensor([r.log['action_idx'] for r in records], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([r.log['log_prob'] for r in records], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([r.log['mask'] for r in records]), dtype=torch.float32, device=self.device)
        
        # Targets
        if 'advantage' not in records[0].log or 'return' not in records[0].log:
            return {}

        advantages = torch.tensor([r.log['advantage'] for r in records], dtype=torch.float32, device=self.device)
        returns = torch.tensor([r.log['return'] for r in records], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Track metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        num_updates = 0

        dataset_size = len(records)
        indices = np.arange(dataset_size)

        # PPO Update Loop
        for _ in range(ppo_epochs):
            self.rng.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Create mini-batch
                mb_probs = probs[batch_indices]
                mb_tables = tables[batch_indices]
                mb_histories = histories[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_masks = masks[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]

                # Create batch state using namedtuple
                batch_state = BatchedState(probabilities=mb_probs, tables=mb_tables, history=mb_histories, legal_actions=mb_masks)
                
                # Forward pass
                outputs = self.network(batch_state)
                
                # Get new log probs and entropy
                card_logits = outputs['card_policy']  # [B, 32]
                
                # Apply mask
                masked_logits = card_logits # Mask is applied inside network
                dist = torch.distributions.Categorical(logits=masked_logits)
                
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                new_values = outputs['value'].squeeze(-1)  # [B]
                
                # Ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Surrogate Loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = nn.functional.mse_loss(new_values, mb_returns)
                
                # Total Loss
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                num_updates += 1

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'total_loss': total_loss / num_updates
        }

    # --- Helper Methods ---

    def _batch_state(self, state: State) -> State:
        """
        Add batch dimension to state tensors.
        Converts Probability matrix to flattened tensor and tables list to tensor.
        """
        # Convert Probability matrix (4, 4, 8) to tensor and flatten to (128,)
        prob_matrix = state.probability.matrix
        prob_tensor = torch.tensor(prob_matrix.flatten(), dtype=torch.float32, device=self.device)
        prob_batch = prob_tensor.unsqueeze(0)  # [1, 128]
        
        # Convert History matrix (4, 4, 8) to tensor and flatten to (128,)
        hist_matrix = state.history.matrix
        hist_tensor = torch.tensor(hist_matrix.flatten(), dtype=torch.float32, device=self.device)
        hist_batch = hist_tensor.unsqueeze(0)  # [1, 128]
        
        # Convert table list to tensor, pad if necessary
        table_list = list(state.table) if state.table else []  # Make a copy!
        # Pad with 36 (unknown card) to make it length 4
        while len(table_list) < 4:
            table_list.append(36)  # padding index
        table_tensor = torch.tensor([int(card) for card in table_list], dtype=torch.long, device=self.device)
        table_batch = table_tensor.unsqueeze(0)  # [1, 4]
        
        # Create a mock State with tensor attributes for network
        state_with_tensors = State(cards=[], trump=state.trump)
        state_with_tensors.probabilities = prob_batch
        state_with_tensors.tables = table_batch
        state_with_tensors.history = hist_batch
        
        return state_with_tensors
