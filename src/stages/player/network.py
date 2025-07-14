import torch
import torch.nn as nn
import torch.nn.functional as F
from src.stages.player.actions import Action

class BeloteNetwork(nn.Module):
    def __init__(self):
        super(BeloteNetwork, self).__init__()
        
        # Constants for Belote
        self.num_ranks = 8  # 7, 8, 9, 10, J, Q, K, A
        self.num_suits = 4  # Spades, Hearts, Diamonds, Clubs
        self.total_actions = self.num_ranks * self.num_suits  # 32 possible cards
        
        # Input sizes
        self.probs_size = 4 * 4 * 8  # 128 - probabilities flattened
        self.table_size = 6  # 6 - table cards: 3 players × 2 values (suit, rank) each
        self.trump_size = 4  # 4 - trump one-hot
        
        # Process each component separately
        self.probs_processor = nn.Sequential(
            nn.Linear(self.probs_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.table_processor = nn.Sequential(
            # nn.Linear(self.table_size, 16),  # Reduced from 32 to 16 since input is much smaller
            nn.Linear(self.table_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.trump_processor = nn.Sequential(
            nn.Linear(self.trump_size, 16),
            nn.ReLU()
        )
        
        # Combine processed features
        self.feature_combiner = nn.Sequential(
            nn.Linear(64 + 16 + 16, 128),  # 96 -> 128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Output heads
        self.card_policy = nn.Linear(64, self.total_actions)
        self.card_value  = nn.Linear(64, 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, action_type, probs_tensor, table_tensor, trump_tensor):
        """
        Forward pass for card actions.
        
        Args:
            action_type: Type of action defined in ./actions.py action type as int
            probs_tensor: Tensor of probabilities [batch, 4, 4, 8] - 4 players x suits x ranks
            table_tensor: Tensor of table cards [batch, 6] - 3 players × 2 values (suit, rank)
            trump_tensor: Tensor of trump suit [batch, 4] - one-hot encoded suit
            
        Returns:
            tuple: (logits, value) for card actions
        """
        # Get features from the shared network
        features = self._extract_features(probs_tensor, table_tensor, trump_tensor)
        
        # Get policy and value for card actions
        if action_type == Action.TYPE_CARD_PLAY:
            # Return logits instead of softmax for proper masking in agent
            card_logits = self.card_policy(features)
            card_value  = self.card_value(features)
            
            return card_logits, card_value
        
        # Future methods for other action types can be added here
        # elif action_type == Action.TYPE_BELOTE:
        #     ...
        
        raise ValueError(f"Unknown action type: {action_type}")

    def _extract_features(self, probs_tensor, table_tensor, trump_tensor):
        """
        Extract features by processing each component separately with linear layers.
        """
        batch_size = probs_tensor.size(0)
        
        # Flatten inputs (table_tensor is already flat with shape [batch, 6])
        probs_flat = probs_tensor.view(batch_size, -1)  # [batch, 128]
        table_flat = table_tensor.view(batch_size, -1)  # [batch, 6] - already flat
        trump_flat = trump_tensor.view(batch_size, -1)  # [batch, 4]
        
        # Process each component separately
        probs_features = self.probs_processor(probs_flat)    # [batch, 64]
        table_features = self.table_processor(table_flat)    # [batch, 16]
        trump_features = self.trump_processor(trump_flat)    # [batch, 16]
        
        # Combine processed features
        combined = torch.cat([probs_features, table_features, trump_features], dim=1)  # [batch, 96]
        
        # Final feature combination
        # features = self.feature_combiner(combined)  # [batch, 64]
        features = table_features
        
        return features