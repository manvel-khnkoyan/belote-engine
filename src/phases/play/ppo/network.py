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

    def __init__(self, state_dim=256, hidden_dim=128, card_emb_dim=16, dropout=0.1):
        super().__init__()

        self.num_cards = 32
        self.card_emb_dim = card_emb_dim
        self.hidden_dim = hidden_dim

        # Card embeddings for table cards
        self.card_embedding = nn.Embedding(37, card_emb_dim, padding_idx=36)

        # ============ ACTION TYPE CLASSIFIER ============
        # Input is Probabilities [B, 128] + History [B, 128] = 256
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
        # Input size matches number of valid cards: 1 card = 16D, 2 cards = 32D, 3 cards = 48D
        self.table_encoder_1card = nn.Sequential(
            nn.Linear(card_emb_dim * 1, hidden_dim),  # 1 card only
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.table_encoder_2card = nn.Sequential(
            nn.Linear(card_emb_dim * 2, hidden_dim),  # 2 cards only
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.table_encoder_3card = nn.Sequential(
            nn.Linear(card_emb_dim * 3, hidden_dim),  # 3 cards only
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
        # Increased complexity: 3 layers
        self.case3_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CASE 4: CLEANUP (3 table cards) - fusion of hand + table[3] + probabilities
        # Maximum complexity: 4 layers
        self.case4_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ============ OUTPUT HEADS ============
        self.card_head = nn.Linear(hidden_dim, self.num_cards)
        self.bid_head = nn.Linear(hidden_dim, 9)
        self.announce_head = nn.Linear(hidden_dim, 10)
        
        # Separate value heads for each case to handle different state representations
        self.value_head_case1 = nn.Linear(hidden_dim, 1)
        self.value_head_case2 = nn.Linear(hidden_dim, 1)
        self.value_head_case3 = nn.Linear(hidden_dim, 1)
        self.value_head_case4 = nn.Linear(hidden_dim, 1)

    def _embed_table(self, tables: torch.Tensor):
        """
        Embed table cards.
        
        Args:
            tables: [B, 4] with card ids 0..31; empty slots are -1
            
        Returns:
            card_emb: [B, 4, card_emb_dim] embeddings (with zeros for empty slots)
            counts: [B] number of valid cards
        """

        # Mark valid cards
        valid = (tables >= 0) & (tables < 32)  # [B, 4]
        idx = tables.clone()
        idx[~valid] = 36  # padding token

        # Embed and zero out empty slots
        card_emb = self.card_embedding(idx)  # [B, 4, card_emb_dim]
        card_emb = card_emb * valid.unsqueeze(-1).float()

        counts = valid.sum(dim=1)  # [B]

        return card_emb, counts

    def forward(self, state: State):
        """
        Forward pass with separate case-specific networks.
        
        Args:
            state: State object with probabilities [B, 128], history [B, 128] and tables [B, 4]
        
        Returns:
            dict with card_policy, bid_policy, announce_policy, value
        """
        probabilities = state.probabilities  # [B, 128]
        history = state.history              # [B, 128]
        tables = state.tables                # [B, 4]

        B = probabilities.size(0)
        device = probabilities.device

        # Combine inputs
        combined_state = torch.cat([probabilities, history], dim=-1) # [B, 256]

        # ============ ACTION TYPE CLASSIFICATION ============
        action_type_logits = self.action_classifier(combined_state.float())  # [B, 2]

        # ============ ENCODE HAND (from probabilities) ============
        hand_encoded = self.prob_encoder(combined_state.float())  # [B, hidden_dim]

        # ============ EMBED TABLE ============
        card_emb, counts = self._embed_table(tables)  # card_emb: [B, 4, card_emb_dim]
        
        # Determine play case based on number of table cards
        stage_counts = torch.clamp(counts, 0, 3)  # [B]

        # Initialize card logits and values
        card_logits = torch.zeros(B, self.num_cards, device=device)
        bid_logits = torch.zeros(B, 9, device=device)
        announce_logits = torch.zeros(B, 10, device=device)
        values = torch.zeros(B, 1, device=device)

        # ============ PROCESS EACH PLAY CASE ============
        cases = [
            {"num_cards": 0, "fusion": self.case1_fusion, "table_encoder": None, "value_head": self.value_head_case1},
            {"num_cards": 1, "fusion": self.case2_fusion, "table_encoder": self.table_encoder_1card, "value_head": self.value_head_case2},
            {"num_cards": 2, "fusion": self.case3_fusion, "table_encoder": self.table_encoder_2card, "value_head": self.value_head_case3},
            {"num_cards": 3, "fusion": self.case4_fusion, "table_encoder": self.table_encoder_3card, "value_head": self.value_head_case4},
        ]

        for case in cases:
            mask = (stage_counts == case["num_cards"])
            if not mask.any():
                continue

            hand = hand_encoded[mask]

            if case["num_cards"] == 0:
                # CASE 1: LEAD - no table cards
                fused = case["fusion"](hand)
            else:
                # CASES 2-4: FOLLOW - concatenate hand + table embeddings
                table_emb = card_emb[mask][:, :case["num_cards"], :].reshape(mask.sum(), -1)
                table_encoded = case["table_encoder"](table_emb)
                fused = case["fusion"](torch.cat([hand, table_encoded], dim=-1))

            card_logits[mask] = self.card_head(fused)
            bid_logits[mask] = self.bid_head(fused)
            announce_logits[mask] = self.announce_head(fused)
            values[mask] = case["value_head"](fused)

        # ============ ACTION MASKING ============
        if hasattr(state, 'legal_actions') and state.legal_actions is not None:
            card_logits = card_logits + state.legal_actions

        # ============ OTHER POLICY HEADS (use hand encoding) ============
        return {
            "action_type": action_type_logits,
            "card_policy": card_logits,  # [B, 32]
            "bid_policy": bid_logits,  # [B, 9]
            "announce_policy": announce_logits,  # [B, 10]
            "value": values,  # [B, 1] - Now uses fused state (Hand + Table)
        }
