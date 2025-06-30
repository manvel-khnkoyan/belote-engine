import torch
import torch.nn as nn
import torch.nn.functional as F
from src.stages.player.actions import Action

class CNNBeloteNetwork(nn.Module):
    def __init__(self):
        super(CNNBeloteNetwork, self).__init__()
        
        # Constants for Belote
        self.num_ranks = 8  # 7, 8, 9, 10, J, Q, K, A
        self.num_suits = 4  # Spades, Hearts, Diamonds, Clubs
        self.total_actions = self.num_ranks * self.num_suits  # 32 possible cards
        
        # Card convolution for hand cards
        # Input will be reshaped from [batch, 4, 4, 8] to [batch, 1, 4, 4, 8] for Conv3d
        self.card_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
        )
        
        # Multi-channel 2D convolution for table cards
        # Process all 3 cards at once as 3 channels
        self.table_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the flattened size of the table features
        # For input [batch, 3, 8, 4] -> Conv2d preserves H,W with padding=1 -> [batch, 32, 8, 4]
        self.table_feature_size = 32 * 8 * 4  # 1024
        
        # Trump info embedding
        self.trump_embedding = nn.Sequential(
            nn.Linear(4, 32),  # Takes a 4-dimensional vector as input
            nn.ReLU()
        )
        
        # Calculate flattened dimensions based on actual conv output
        # After 3D convolutions: [batch, 128, 4, 4, 8] -> flattened: 128 * 4 * 4 * 8 = 16384
        self.player_flat_dim = 128 * 4 * 4 * 8  # 16384 (corrected!)
        
        # Feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(self.player_flat_dim + self.table_feature_size + 32, 512),  # 16384 + 1024 + 32 = 17440
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # MULTIPLE HEADS (one for each action type)
        
        # Play card - Action type 1
        self.card_policy = nn.Linear(256, self.total_actions)
        self.card_value = nn.Linear(256, 1)

        # Belote/Rebelote policy - Action type 2
        # self.belote_policy = nn.Linear(256, 2)  # Yes/No decision
        # self.belote_value = nn.Linear(256, 1)


    def forward(self, action_type, probs_tensor, table_tensor, trump_tensor):
        """
        Forward pass for 'card' action type (default).
        
        Args:
            action_type: Type of action defined in ./actions.py action type as int
            probs_tensor: Tensor of probabilities [batch, 4, 4, 8] - 4 players x suits x ranks
            table_tensor: Tensor of table cards [batch, 3, 8, 4] - 3 cards x ranks x suits  
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
            card_value = self.card_value(features)
            
            return card_logits, card_value
        
        # Future methods for other action types
        # ......

        raise ValueError(f"Unknown action type: {action_type}")

    
    def _extract_features(self, probs_tensor, table_tensor, trump_tensor):
        """
        Extract features from the shared network layers.
        This allows code reuse across different action types.
        """
        # Get batch size from the input tensor
        batch_size = probs_tensor.size(0)
        
        # Reshape probs_tensor for 3D convolution: [batch, 4, 4, 8] -> [batch, 1, 4, 4, 8]
        probs_tensor_3d = probs_tensor.unsqueeze(1)
        
        # Process each player separately through their own conv layers
        probs_features = self.card_conv(probs_tensor_3d)
        probs_features = probs_features.view(batch_size, -1)  # Flatten
        
        # Process table cards
        table_features = self.table_conv(table_tensor)
        
        # Trump processing
        trump_features = self.trump_embedding(trump_tensor)
        
        # Combine all features
        combined = torch.cat([probs_features, table_features, trump_features], dim=1)
        shared_out = self.shared_fc(combined)
        
        return shared_out