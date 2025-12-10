import torch
import torch.nn as nn
import torch.nn.functional as F

class PPONetwork(nn.Module):
    def __init__(self, state_dim=128, card_emb_dim=16, hidden_dim=128):
        super(PPONetwork, self).__init__()
        
        # Constants
        self.num_cards = 32
        self.num_ranks = 8
        self.card_emb_dim = card_emb_dim
        
        # --- 1. Embeddings ---
        # 0-31: Regular cards
        # 32-35: Trump Suit Modifiers (Spades, Hearts, Diamonds, Clubs)
        # 36: Padding/Empty slot
        # We use an additive embedding approach: Card Embedding + Trump Suit Modifier (if trump)
        self.card_embedding = nn.Embedding(37, card_emb_dim, padding_idx=36)
        
        # --- 2. Input Encoders ---
        
        # Cards Held Encoder
        # Input: 32 (Multi-hot vector of cards in hand)
        self.hand_encoder = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Shared State Encoder
        # Encodes the game history (probabilities) and trump information
        # Input: state_dim (128) + trump_info (5)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # --- 3. Stage-Specific Layers (The "Genius" Architecture) ---
        # We separate the network into 4 distinct branches based on the number of cards on the table (0, 1, 2, 3).
        # This allows the agent to learn distinct strategies for Leading, Second, Third, and Last positions.
        self.stage_layers = nn.ModuleList([
            nn.Sequential(
                # Input: Hand Features + Shared Features + Flattened Table Embeddings (i * card_emb_dim)
                nn.Linear(hidden_dim * 2 + (i * card_emb_dim), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for i in range(4)
        ])
        
        # --- 4. Action Heads ---
        # Separate Policy Heads for each stage to further specialize the decision making
        self.play_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_cards) for _ in range(4)
        ])
        
        # Shared Heads for other tasks
        # Using combined features (Hand + State)
        self.bid_head = nn.Linear(hidden_dim * 2, 9)       # Assuming 9 bid types
        self.announce_head = nn.Linear(hidden_dim * 2, 10) # Assuming 10 announce types
        self.value_head = nn.Linear(hidden_dim * 2, 1)     # Critic

        self._init_weights()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _get_card_embeddings(self, tables, trumps):
        """
        Computes embeddings by adding a 'Trump Suit Modifier' to the base card embedding
        if the card is a trump card.
        """
        # 1. Base Embeddings
        valid_mask = (tables >= 0) & (tables < 32)
        base_indices = tables.clone()
        base_indices[~valid_mask] = 36 # Padding
        base_embeddings = self.card_embedding(base_indices)
        
        # 2. Identify Trump Cards
        # trumps: [B, 5] -> [Mode, S0, S1, S2, S3]
        is_regular_mode = (trumps[:, 0] < 0.5)
        trump_suits = torch.argmax(trumps[:, 1:], dim=1) # [B]
        
        trump_suits_exp = trump_suits.unsqueeze(1).expand_as(tables)
        is_regular_mode_exp = is_regular_mode.unsqueeze(1).expand_as(tables)
        
        card_suits = torch.div(tables, self.num_ranks, rounding_mode='floor')
        is_trump_suit = (card_suits == trump_suits_exp)
        is_trump_card = is_regular_mode_exp & is_trump_suit & valid_mask
        
        # 3. Trump Modifiers
        # Indices 32-35 correspond to Spades, Hearts, Diamonds, Clubs modifiers
        safe_suits = card_suits.clone()
        safe_suits[~valid_mask] = 0 # Avoid out of bounds
        modifier_indices = 32 + safe_suits
        
        modifier_embeddings = self.card_embedding(modifier_indices)
        
        # Add modifier only where card is trump
        final_embeddings = base_embeddings + (modifier_embeddings * is_trump_card.unsqueeze(-1).float())
        
        return final_embeddings, valid_mask

    def forward(self, hand, probabilities, tables, trumps):
        """
        hand: [B, 32] - Multi-hot vector of cards in hand
        probabilities: [B, 128] - Game state/history
        tables: [B, 4] - Card IDs (0-31) on table. Use -1 or 64 for empty slots.
        trumps: [B, 5] - Trump encoding
        """
        B = probabilities.shape[0]
        device = probabilities.device
        
        # 1. Encode Inputs
        hand_features = self.hand_encoder(hand) # [B, H]
        
        state_input = torch.cat([probabilities, trumps], dim=1)
        shared_features = self.state_encoder(state_input) # [B, H]
        
        # 2. Process Table Cards
        table_embeddings, valid_mask = self._get_card_embeddings(tables, trumps)
        
        # 3. Route by Table Count (The "Genius" Branching)
        # Count valid cards to determine the stage (0, 1, 2, or 3 cards on table)
        counts = valid_mask.sum(dim=1) # [B]
        
        card_logits = torch.zeros(B, self.num_cards, device=device)
        
        # Iterate over stages and process relevant batch items
        for stage in range(4):
            mask = (counts == stage)
            if not mask.any():
                continue
                
            # Filter batch for this stage
            sub_hand = hand_features[mask]
            sub_state = shared_features[mask]
            
            if stage == 0:
                # No cards on table
                stage_input = torch.cat([sub_hand, sub_state], dim=1)
            else:
                # Flatten the first 'stage' cards
                sub_table = table_embeddings[mask, :stage].reshape(mask.sum(), -1)
                stage_input = torch.cat([sub_hand, sub_state, sub_table], dim=1)
            
            # Pass through specialized layer and head
            out = self.stage_layers[stage](stage_input)
            logits = self.play_heads[stage](out)
            
            card_logits[mask] = logits
            
        # 4. Other Heads (Shared Context)
        combined_features = torch.cat([hand_features, shared_features], dim=1)
        
        bid_logits = self.bid_head(combined_features)
        announce_logits = self.announce_head(combined_features)
        value = self.value_head(combined_features)
        
        return {
            'card_policy': card_logits,
            'bid_policy': bid_logits,
            'announce_policy': announce_logits,
            'value': value
        }
