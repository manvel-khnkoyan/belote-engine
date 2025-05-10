import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBeloteNetwork(nn.Module):
    def __init__(self):
        super(CNNBeloteNetwork, self).__init__()
        
        # Constants for Belote
        self.num_ranks = 8  # 7, 8, 9, 10, J, Q, K, A
        self.num_suits = 4  # Spades, Hearts, Diamonds, Clubs
        self.total_actions = self.num_ranks * self.num_suits  # 32 possible cards
        
        # Card convolution for hand cards
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
        self.table_feature_size = 32 * 8 * 4  # Size after Conv2d layers
        
        # Trump info embedding
        self.trump_embedding = nn.Sequential(
            nn.Linear(4, 32),  # Takes a 4-dimensional vector as input
            nn.ReLU()
        )
        
        # Calculate flattened dimensions
        self.card_flat_dim = 128 * 4 * 8 * 4  # Assuming 4 players, 8 ranks, 4 suits
        
        # Feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(self.card_flat_dim + self.table_feature_size + 32, 512),  # Hand + Table + Trump
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # HAEDS

        # Play card
        self.play_policy = nn.Linear(256, self.total_actions)
        self.play_value = nn.Linear(256, 1)

        # Belote/Rebelote policy
        # ...


    def forward(self, probs_tensor, table_tensor, trump_tensor):
        """
        Forward pass through the network with NO shape checks.
        
        Expected input shapes for card_conv:
        - probs_tensor must be [batch, 1, 4, 4, 8] for Conv3D layer
        """
        # Get batch size from the input tensor
        batch_size = probs_tensor.size(0)
        
        # Process through convolutional layers - NO SHAPE MANIPULATION
        probs_features = self.card_conv(probs_tensor)
        probs_features = probs_features.view(batch_size, -1)  # Flatten
        
        # Process table cards
        table_features = self.table_conv(table_tensor)
        
        # Trump processing
        trump_features = self.trump_embedding(trump_tensor)
        
        # Combine all features
        combined = torch.cat([probs_features, table_features, trump_features], dim=1)
        shared_out = self.shared_fc(combined)
        
        # Outputs
        play_policy = F.softmax(self.policy_head(shared_out), dim=-1)
        play_value = self.value_head(shared_out)
        
        return play_policy, play_value