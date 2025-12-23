import torch
import torch.nn as nn
from src.phases.play.core.state import State


class PPONetwork(nn.Module):
    """
    Action-Specific Network with Separate Play Cases.
    
    INPUTS: Probabilities [B, 128] (Game History)
    
    ACTION TYPE CLASSIFIER: Probabilities → Action Type (play, declare, etc.)
    
    PLAY ACTION NETWORK:
    ├── CASE 1: LEAD (0 table cards)
    │   └─ Fusion(Hand, Probabilities) → Card Policy
    ├── CASE 2: FOLLOW (1 table card)
    │   └─ Fusion(Hand, Table_Emb[1], Probabilities) → Card Policy
    ├── CASE 3: FOLLOW (2 table cards)
    │   └─ Fusion(Hand, Table_Emb[2], Probabilities) → Card Policy
    └── CASE 4: CLEANUP (3 table cards)
        └─ Fusion(Hand, Table_Emb[3], Probabilities) → Card Policy
    """

    def __init__(self, state_dim=128, hidden_dim=128, card_emb_dim=16, dropout=0.1):
        super().__init__()

        self.num_cards = 32
        self.card_emb_dim = card_emb_dim
        self.hidden_dim = hidden_dim

        # Card embeddings for table cards
        self.card_embedding = nn.Embedding(37, card_emb_dim, padding_idx=36)

        # ============ ACTION TYPE CLASSIFIER ============
        self.action_classifier = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 0: play, 1: declare (future)
        )

        # ============ PROBABILITY ENCODER (Hand representation) ============
        self.prob_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ============ TABLE ENCODERS (for each play case) ============
        # Each case has its own table encoder for the specific number of cards
        self.table_encoder_1card = nn.Sequential(
            nn.Linear(card_emb_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.table_encoder_2card = nn.Sequential(
            nn.Linear(card_emb_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.table_encoder_3card = nn.Sequential(
            nn.Linear(card_emb_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ============ CASE-SPECIFIC FUSION NETWORKS ============
        # CASE 1: LEAD (0 table cards) - fusion of hand + probabilities only
        self.case1_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CASE 2: FOLLOW (1 table card) - fusion of hand + table[1] + probabilities
        self.case2_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CASE 3: FOLLOW (2 table cards) - fusion of hand + table[2] + probabilities
        self.case3_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CASE 4: CLEANUP (3 table cards) - fusion of hand + table[3] + probabilities
        self.case4_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ============ OUTPUT HEADS ============
        self.card_head = nn.Linear(hidden_dim, self.num_cards)
        self.bid_head = nn.Linear(hidden_dim, 9)
        self.announce_head = nn.Linear(hidden_dim, 10)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _embed_table(self, tables: torch.Tensor):
        """
        Embed table cards.
        
        Args:
            tables: [B, 4] with card ids 0..31; empty slots are -1
            
        Returns:
            table_flat: [B, 4*card_emb_dim]
            counts: [B] number of valid cards
        """
        B = tables.size(0)

        # Mark valid cards
        valid = (tables >= 0) & (tables < 32)  # [B, 4]
        idx = tables.clone()
        idx[~valid] = 36  # padding token

        # Embed and zero out empty slots
        card_emb = self.card_embedding(idx)  # [B, 4, card_emb_dim]
        card_emb = card_emb * valid.unsqueeze(-1).float()

        # Flatten
        table_flat = card_emb.reshape(B, -1)  # [B, 4*card_emb_dim]
        counts = valid.sum(dim=1)  # [B]

        return table_flat, counts

    def forward(self, state: State):
        """
        Forward pass with separate case-specific networks.
        
        Args:
            state: State object with probabilities [B, 128] and tables [B, 4]
        
        Returns:
            dict with card_policy, bid_policy, announce_policy, value
        """
        probabilities = state.probabilities  # [B, 128]
        tables = state.tables  # [B, 4]

        B = probabilities.size(0)
        device = probabilities.device

        # ============ ACTION TYPE CLASSIFICATION ============
        self.action_classifier(probabilities.float())  # [B, 2]

        # ============ ENCODE HAND (from probabilities) ============
        hand_encoded = self.prob_encoder(probabilities.float())  # [B, hidden_dim]

        # ============ EMBED TABLE ============
        table_flat, counts = self._embed_table(tables)  # table_flat: [B, 4*card_emb_dim]
        
        # Determine play case based on number of table cards
        stage_counts = torch.clamp(counts, 0, 3)  # [B]

        # Initialize card logits
        card_logits = torch.zeros(B, self.num_cards, device=device)

        # ============ CASE 1: LEAD (0 table cards) ============
        case1_mask = (stage_counts == 0)
        if case1_mask.any():
            hand_case1 = hand_encoded[case1_mask]  # [b, hidden_dim]
            fused_case1 = self.case1_fusion(hand_case1)
            card_logits[case1_mask] = self.card_head(fused_case1)

        # ============ CASE 2: FOLLOW (1 table card) ============
        case2_mask = (stage_counts == 1)
        if case2_mask.any():
            hand_case2 = hand_encoded[case2_mask]  # [b, hidden_dim]
            table_case2 = self.table_encoder_1card(table_flat[case2_mask])  # [b, hidden_dim]
            fused_input_case2 = torch.cat([hand_case2, table_case2], dim=-1)  # [b, 2*hidden_dim]
            fused_case2 = self.case2_fusion(fused_input_case2)
            card_logits[case2_mask] = self.card_head(fused_case2)

        # ============ CASE 3: FOLLOW (2 table cards) ============
        case3_mask = (stage_counts == 2)
        if case3_mask.any():
            hand_case3 = hand_encoded[case3_mask]  # [b, hidden_dim]
            table_case3 = self.table_encoder_2card(table_flat[case3_mask])  # [b, hidden_dim]
            fused_input_case3 = torch.cat([hand_case3, table_case3], dim=-1)  # [b, 2*hidden_dim]
            fused_case3 = self.case3_fusion(fused_input_case3)
            card_logits[case3_mask] = self.card_head(fused_case3)

        # ============ CASE 4: CLEANUP (3 table cards) ============
        case4_mask = (stage_counts == 3)
        if case4_mask.any():
            hand_case4 = hand_encoded[case4_mask]  # [b, hidden_dim]
            table_case4 = self.table_encoder_3card(table_flat[case4_mask])  # [b, hidden_dim]
            fused_input_case4 = torch.cat([hand_case4, table_case4], dim=-1)  # [b, 2*hidden_dim]
            fused_case4 = self.case4_fusion(fused_input_case4)
            card_logits[case4_mask] = self.card_head(fused_case4)

        # ============ OTHER POLICY HEADS (use hand encoding) ============
        return {
            "card_policy": card_logits,  # [B, 32]
            "bid_policy": self.bid_head(hand_encoded),  # [B, 9]
            "announce_policy": self.announce_head(hand_encoded),  # [B, 10]
            "value": self.value_head(hand_encoded),  # [B, 1]
        }
