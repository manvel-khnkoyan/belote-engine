import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming these imports are available in your project structure
from src.stages.player.actions import ActionPlayCard, ActionBid, ActionAnnounce # Added Bid and Announce actions for multi-head
from src.states.probability import Probability
from src.states.table import Table
from src.trump import Trump
from src.card import Card # Added Card import for _card_to_embedding_id

class PPONetwork(nn.Module):
    def __init__(self):
        super(PPONetwork, self).__init__()

        # Constants for Belote
        self.num_ranks = 8  # 7, 8, 9, 10, J, Q, K, A
        self.num_suits = 4  # Spades, Hearts, Diamonds, Clubs

        self.total_card_types = self.num_ranks * self.num_suits # 32 unique cards
        self.card_embedding_dim = 16 # Dimension for card embeddings

        # Input sizes
        # Probability matrix is (4 players, 4 suits, 8 ranks)
        self.probs_size = 4 * self.num_suits * self.num_ranks # 128

        # Trump encoding: mode (1 bit) + suit (4 bits one-hot, only used when mode=REGULAR)
        self.trump_size = 1 + self.num_suits # 5 total (1 for mode + 4 for suit)
        
        # Bid encoding: (Pass, RegularTrump, NoTrump, AllTrump, Coeur, Carreau, Trefle, Pique)
        # Assuming 8 types of bids (Pass, General contracts for each suit, No Trump, All Trump)
        # This will depend on your specific ActionBid implementation.
        # For simplicity, let's assume a fixed number of bid options, e.g., 9 (Pass + 8 contract types)
        self.total_bid_types = 9 
        
        # Announce encoding: (Belote/Rebelote, Carre, Tierce, Quarte, Quinte)
        # This will also depend on your ActionAnnounce implementation.
        # Let's assume a fixed number, e.g., 5-10
        self.total_announce_types = 10 # Example, adjust as needed

        # --- Card Embedding Layer ---
        # Card states depend on trump mode:
        # - NO_TRUMP: just the card itself (32 possibilities)
        # - REGULAR: card + whether it's trump (32 * 2 = 64 possibilities)
        # We'll use the larger space to handle both cases
        self.card_embedding = nn.Embedding(2 * self.total_card_types, self.card_embedding_dim)

        # --- Shared Feature Extractor for all heads ---
        # This part will be executed regardless of the action type
        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(self.probs_size + self.trump_size, 128), # Combined with trump
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # --- Stage-specific components for Table Cards (optional, if you want different processing) ---
        # For simplicity, we can have the shared_feature_extractor handle base features,
        # and then concatenate table cards and pass through stage-specific layers.
        # The existing stage_networks approach is good. We'll modify it slightly.
        self.stage_networks = nn.ModuleList()
        for i in range(4): # For 0, 1, 2, 3 cards on table
            # Input to stage_network will be (64 from shared_feature_extractor + i * card_embedding_dim)
            input_dim = 64 + i * self.card_embedding_dim
            self.stage_networks.append(
                nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64), # Final feature size before heads
                    nn.ReLU()
                )
            )

        # --- Output heads ---
        # Each policy head will output logits for its respective action space
        self.card_policy_head    = nn.Linear(64, self.total_card_types)
        self.bid_policy_head     = nn.Linear(64, self.total_bid_types)
        self.announce_policy_head = nn.Linear(64, self.total_announce_types)
        
        # Value head (predicting game outcome/value) - common for all states
        self.value_head          = nn.Linear(64, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _trump_to_tensor(self, trump: Trump):
        """Convert Trump object to tensor representation"""
        # Create tensor: [mode, suit_0, suit_1, suit_2, suit_3]
        trump_tensor = torch.zeros(self.trump_size, device=next(self.parameters()).device) # Ensure on same device
        
        # Set mode (0 for REGULAR, 1 for NO_TRUMP) - assuming REGULAR=0, NO_TRUMP=1
        # Adjust based on your Trump.py constants if different
        trump_tensor[0] = 1.0 if trump.mode == Trump.NO_TRUMP else 0.0
        
        # Set suit one-hot encoding (only if REGULAR mode)
        if trump.mode == Trump.REGULAR and trump.suit is not None:
            trump_tensor[1 + trump.suit] = 1.0
            
        return trump_tensor

    def _is_card_trump(self, card: Card, trump: Trump):
        """Determine if a card is trump based on the Trump object"""
        if trump.mode == Trump.NO_TRUMP:
            return False
        elif trump.mode == Trump.REGULAR:
            return card.suit == trump.suit if trump.suit is not None else False
        return False

    def _card_to_embedding_id(self, card: Card, trump: Trump):
        """Convert card to embedding ID based on trump context"""
        # Base card ID (0-31)
        base_id = card.suit * self.num_ranks + card.rank
        
        # If NO_TRUMP mode, use base_id directly
        if trump.mode == Trump.NO_TRUMP:
            return base_id
        
        # If REGULAR mode, add trump information
        is_trump = self._is_card_trump(card, trump)
        # Trump cards get an offset to differentiate them from non-trump cards of the same rank/suit
        return base_id + (self.total_card_types if is_trump else 0)

    def forward(self, probabilities, tables, trumps):
        # Handle batch or single input
        is_batched = isinstance(probabilities, torch.Tensor) and probabilities.dim() == 4
        
        if not is_batched:
            probabilities = probabilities.to_tensor().unsqueeze(0).to(next(self.parameters()).device)
            tables = tables.unsqueeze(0).to(next(self.parameters()).device) # Assuming Table.to_tensor() is implemented similarly
            trumps = self._trump_to_tensor(trumps).unsqueeze(0)
        else:
            # Ensure all inputs are on the correct device
            probabilities = probabilities.to(next(self.parameters()).device)
            # Assuming tables and trumps are already tensors in batch mode
            tables = tables.to(next(self.parameters()).device)
            trumps = trumps.to(next(self.parameters()).device)


        batch_size = probabilities.size(0)

        # Reshape probability tensor for linear layer
        probs_flat = probabilities.view(batch_size, -1) # [batch, 128]

        # Concatenate trump information
        combined_probs_trump = torch.cat([probs_flat, trumps], dim=1) # [batch, 128 + 5]

        # Shared feature extraction
        shared_features = self.shared_feature_extractor(combined_probs_trump) # [batch, 64]

        # Determine the current stage based on the number of cards on the table
        # If batching, we need to handle variable table sizes per batch item.
        # For simplicity, let's assume all items in a batch are from the same "stage" (same number of table cards).
        # This requires careful batching in PPOMemory or during sample creation.
        
        # For now, let's assume single input or a batch with uniform table_cards_count
        table_cards_count = tables.shape[1] if is_batched else tables.index # Assuming table.index for single, tables.shape[1] for batched
        
        # Stage-specific processing for table cards
        if table_cards_count > 0:
            # table is (batch_size, num_cards_on_table, card_encoding_dim) if already numeric
            # If table is list of Table objects, need to process each:
            
            # Let's assume `tables` input to forward is already a tensor representing card IDs or features
            # for simplicity in this general network code.
            # If tables contains Card objects, this part needs to convert them.
            
            # Assuming `tables` is a tensor of card IDs, shape (batch_size, num_cards_on_table)
            # We need to compute embeddings for each card based on trump context.
            
            # This is complex with heterogeneous batching (different num cards on table).
            # Simplest approach: Assume `tables` comes in as already embedded features or card IDs that
            # are then embedded, and the number of cards on table is fixed per batch.
            
            # For current state, we'll assume `tables` is a pre-processed tensor of embedded card features.
            # OR, if it's a list of Table objects, we'd need to loop:
            
            # Since _card_to_embedding_id expects a Card object and a Trump object,
            # if `tables` is a batch of `Table` objects, we need to adapt.
            
            # A more robust way: `tables` input to network is a list of Card objects or a tensor of card_embedding_ids
            # For simplicity, let's assume `tables` is a tensor of card IDs that we embed.
            # The `table.cards` structure suggests processing Card objects.
            
            embedded_table_cards_list = []
            if is_batched:
                # If `tables` input is a tensor of numerical card IDs already
                # shape (batch_size, table_cards_count)
                # Need to convert these IDs to the full embedding_id considering trump.
                # This requires knowing original card + trump.
                # This is a key point where the network input needs careful design.
                # For now, let's assume a simplified scenario where tables input is a tensor
                # of (suit*rank) and we append trump info for embedding lookup.
                
                # A robust solution might involve another 'transformer' function for tables
                # similar to SuitsCanonicalTransformer
                
                # For now, let's assume the input `tables` is a tensor of (suit*rank) IDs
                # and we need to determine trump status for embedding.
                # This means `trump` also needs to be batched.
                
                # The provided ppo_agent.py implies `tables` is a Table object, and `trumps` is a Trump object.
                # We need to correctly convert them to tensors *before* calling network.forward.
                # This means the transformation logic should largely be in the agent's act/learn methods.
                
                # Let's revert to processing a single Table object as implied by the original code,
                # then consider batching `Table` objects' features later.
                # In forward, `tables` and `trumps` are *already* tensors.
                # If they are, then `table_cards_count` from `tables.shape[1]` is already the number of embedded card slots.
                
                # If `tables` is a tensor of embedded card IDs:
                table_card_ids = tables.long() # Assuming tables is (batch_size, num_cards_on_table) tensor of card IDs (0-31)
                
                # This part is tricky. The card embedding depends on `is_trump`.
                # If we assume `tables` contains a pre-computed embedding ID (0-63)
                # that already encodes trump status, then:
                table_card_embeddings = self.card_embedding(table_card_ids) # [batch, N, card_embedding_dim]
                
                table_card_features = table_card_embeddings.view(batch_size, -1) # [batch, N * card_embedding_dim]
                
                combined_features = torch.cat([shared_features, table_card_features], dim=1)
            else: # Single input case - `tables` is a Table object, `trumps` is a Trump object
                card_embeddings_list = []
                for card in tables.cards: # Iterate over Card objects in the Table
                    if card is not None:
                        card_id = self._card_to_embedding_id(card, trumps)
                        card_embeddings_list.append(self.card_embedding(torch.tensor(card_id, device=shared_features.device).unsqueeze(0)))
                
                if card_embeddings_list:
                    table_card_embeddings = torch.cat(card_embeddings_list, dim=1) # Concatenate embeddings
                    table_card_features = table_card_embeddings.view(batch_size, -1) # Flatten
                    combined_features = torch.cat([shared_features, table_card_features], dim=1)
                else:
                    combined_features = shared_features
        else:
            combined_features = shared_features # Only probabilities + trump when no cards on table

        # Select the appropriate stage network
        # This still implies that table_cards_count is uniform across batch if batched.
        # This is an assumption you need to manage during batching if you use varying table sizes.
        
        # If `tables` input is a tensor of pre-embedded features, then the stage_networks would need to adapt to input_dim.
        # For now, let's use a single `final_feature_layer` that takes `combined_features`.
        # This simplifies the stage_networks part considerably.
        
        # New approach: remove stage_networks and use one final feature layer after combining.
        # This implies a fixed input size to final_feature_layer, which means
        # table cards need to be padded if their count varies.
        
        # Let's keep `stage_networks` but fix the `tables` input representation
        # in the `act` and `learn` methods of PPOAgent to be consistent.
        
        # `table_cards_count` for stage_networks needs to be the number of cards that were actually embedded.
        # If `combined_features` already handles it by padding, then `table_cards_count` refers to slots.
        
        # Assuming table_cards_count determines which `stage_networks` to use.
        # This is where batching is tricky for variable length sequences.
        # For now, if batched, `table_cards_count` should be inferred from the second dim of `tables`.
        
        # To make it truly multi-head and allow flexible table states,
        # we can feed `combined_features` into a final shared layer, then to heads.
        
        # Let's assume the feature computation for table cards results in a fixed-size vector
        # by padding or summing if variable number of cards.
        
        # Simplification: A single final shared processing layer.
        final_features_pre_heads = nn.Sequential(
            nn.Linear(combined_features.size(-1), 64), # Adapt input dim
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(combined_features.device)(combined_features) # Ensure on same device

        # --- Output Heads ---
        # All heads receive the same `final_features_pre_heads`
        card_logits    = self.card_policy_head(final_features_pre_heads)    # [batch, 32]
        bid_logits     = self.bid_policy_head(final_features_pre_heads)     # [batch, num_bid_types]
        announce_logits = self.announce_policy_head(final_features_pre_heads) # [batch, num_announce_types]
        
        value          = self.value_head(final_features_pre_heads)          # [batch, 1]

        # Return a dictionary of all possible outputs.
        # The agent will select the relevant ones.
        return {
            'card_policy': card_logits,
            'bid_policy': bid_logits,
            'announce_policy': announce_logits,
            'value': value
        }