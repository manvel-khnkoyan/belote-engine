import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from src.phases.play.core.agent import Agent
from src.phases.play.core.state import State
from src.phases.play.core.actions import Action, ActionPlayCard
from src.phases.play.core.record import Record
from src.models.trump import TrumpMode
from src.phases.play.ppo.network import PPONetwork

class PpoAgent(Agent):
    def __init__(self, network: PPONetwork):
        self.network = network
        self.device = next(network.parameters()).device
        self.optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Dict[str, Any] | None]:
        """
        Choose an action based on the current game state.
        """
        # 1. Prepare Inputs
        hand_tensor = self._prepare_hand(state.cards)
        prob_tensor = self._prepare_probability(state.probability)
        table_tensor = self._prepare_table(state.table)
        trump_tensor = self._prepare_trump(state.trump)

        # 2. Forward Pass
        # Add batch dimension [1, ...]
        with torch.no_grad():
            outputs = self.network(
                hand_tensor.unsqueeze(0), 
                prob_tensor.unsqueeze(0), 
                table_tensor.unsqueeze(0), 
                trump_tensor.unsqueeze(0)
            )
        
        # 3. Mask Invalid Actions
        # We only handle ActionPlayCard for the network's card_policy head
        card_logits = outputs['card_policy'][0] # [32]
        
        valid_indices = []
        action_map = {}
        
        # Filter for PlayCard actions
        play_card_actions = [a for a in actions if isinstance(a, ActionPlayCard)]
        
        if not play_card_actions:
            # If no card play actions (e.g. only Pass?), pick the first available action
            # and return no log (not training on non-play actions yet)
            return actions[0], None

        for action in play_card_actions:
            idx = action.card.suit * 8 + action.card.rank
            valid_indices.append(idx)
            action_map[idx] = action
            
        mask = torch.full_like(card_logits, -float('inf'))
        for idx in valid_indices:
            mask[idx] = 0
            
        masked_logits = card_logits + mask
        probs = torch.softmax(masked_logits, dim=0)
        
        # 4. Sample Action
        dist = torch.distributions.Categorical(probs)
        action_idx_tensor = dist.sample()
        action_idx = action_idx_tensor.item()
        
        chosen_action = action_map[action_idx]
        
        # 5. Prepare Logs
        log_prob = dist.log_prob(action_idx_tensor)
        value = outputs['value'][0]
        
        log = {
            'log_prob': log_prob.item(),
            'value': value.item(),
            'state': {
                'hand': hand_tensor.cpu().numpy(),
                'probability': prob_tensor.cpu().numpy(),
                'table': table_tensor.cpu().numpy(),
                'trump': trump_tensor.cpu().numpy()
            },
            'action_idx': action_idx,
            'mask': mask.cpu().numpy()
        }
        
        return chosen_action, log

    def learn(self, records: List[Record]):
        """
        Train the agent based on the collected records.
        Assumes record.log contains 'advantage' and 'return' keys computed by the caller.
        """
        if not records:
            return

        # Hyperparameters
        clip_param = 0.2
        value_loss_coef = 0.5
        entropy_coef = 0.01
        max_grad_norm = 0.5
        ppo_epochs = 4
        
        # 1. Collate data into tensors
        # States
        hands = torch.tensor(np.array([r.log['state']['hand'] for r in records]), dtype=torch.float32, device=self.device)
        probs = torch.tensor(np.array([r.log['state']['probability'] for r in records]), dtype=torch.float32, device=self.device)
        tables = torch.tensor(np.array([r.log['state']['table'] for r in records]), dtype=torch.long, device=self.device)
        trumps = torch.tensor(np.array([r.log['state']['trump'] for r in records]), dtype=torch.float32, device=self.device)
        
        # Actions & Old Log Probs
        actions = torch.tensor([r.log['action_idx'] for r in records], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([r.log['log_prob'] for r in records], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([r.log['mask'] for r in records]), dtype=torch.float32, device=self.device)
        
        # Targets (Assumed to be present in logs)
        if 'advantage' not in records[0].log or 'return' not in records[0].log:
            # If advantages are not pre-calculated, we cannot proceed with PPO update in this structure
            # For now, we assume the caller handles GAE or return calculation and augments the logs
            return

        advantages = torch.tensor([r.log['advantage'] for r in records], dtype=torch.float32, device=self.device)
        returns = torch.tensor([r.log['return'] for r in records], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. PPO Update Loop
        for _ in range(ppo_epochs):
            # Forward pass
            outputs = self.network(hands, probs, tables, trumps)
            
            # Get new log probs and entropy
            card_logits = outputs['card_policy'] # [B, 32]
            
            # Apply mask
            masked_logits = card_logits + masks
            dist = torch.distributions.Categorical(logits=masked_logits)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            new_values = outputs['value'].squeeze(-1) # [B]
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value Loss
            value_loss = nn.functional.mse_loss(new_values, returns)
            
            # Total Loss
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
            self.optimizer.step()

    # --- Helper Methods ---

    def _prepare_hand(self, cards):
        hand_vec = torch.zeros(32, dtype=torch.float32, device=self.device)
        for card in cards:
            idx = card.suit * 8 + card.rank
            hand_vec[idx] = 1.0
        return hand_vec

    def _prepare_probability(self, probability):
        # Flatten the [4, 4, 8] matrix to [128]
        prob_np = probability.matrix.flatten()
        return torch.tensor(prob_np, dtype=torch.float32, device=self.device)

    def _prepare_table(self, table_cards):
        # Convert table cards to indices [4]
        # 0-31 for cards, 36 for empty/padding
        table_indices = []
        for card in table_cards:
            idx = card.suit * 8 + card.rank
            table_indices.append(idx)
        
        # Pad to 4
        while len(table_indices) < 4:
            table_indices.append(36)
            
        return torch.tensor(table_indices, dtype=torch.long, device=self.device)

    def _prepare_trump(self, trump):
        # Encode trump [5]
        # [Mode, S0, S1, S2, S3]
        # Mode: 0 for Regular, 1 for AllTrump/NoTrump (simplified)
        # Actually network expects: is_regular_mode = (trumps[:, 0] < 0.5)
        
        trump_vec = [0.0] * 5
        if trump.mode == TrumpMode.Regular:
            trump_vec[0] = 0.0 # Regular mode
            if trump.suit is not None:
                trump_vec[1 + trump.suit] = 1.0
        else:
            trump_vec[0] = 1.0 # Non-regular mode (AllTrump/NoTrump)
            
        return torch.tensor(trump_vec, dtype=torch.float32, device=self.device)
