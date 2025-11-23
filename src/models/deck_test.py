import pytest
from unittest.mock import patch
from deck import Deck
from card import Card

class TestDeck:
    def test_initialization(self):
        d = Deck()
        assert len(d.cards) == 32
        
        # Check distribution
        suits = [0] * 4
        ranks = [0] * 8
        for c in d.cards:
            suits[c.suit] += 1
            ranks[c.rank] += 1
            
        assert all(s == 8 for s in suits)
        assert all(r == 4 for r in ranks)

    def test_deal(self):
        d = Deck()
        hands = d.deal()
        assert len(hands) == 4
        assert all(len(h) == 8 for h in hands)
        
        # Verify all cards are present and unique across hands
        all_cards = []
        for h in hands:
            all_cards.extend(h)
        
        assert len(all_cards) == 32
        assert len(set((c.suit, c.rank) for c in all_cards)) == 32

    def test_transform_suits_canonical_structure(self):
        d = Deck()
        # Save original suits
        original_suits = [c.suit for c in d.cards]
        
        transform, reverse = d.transform_suits_canonical()
        
        # Check that cards were modified
        new_suits = [c.suit for c in d.cards]
        
        # Check that transform and reverse are consistent
        for s in range(4):
            canon = transform(s)
            orig = reverse(canon)
            assert orig == s
            
        # Check that the modification on cards matches the transform
        for i, orig_suit in enumerate(original_suits):
            assert new_suits[i] == transform(orig_suit)

    @patch('random.shuffle')
    def test_transform_suits_canonical_logic(self, mock_shuffle):
        # We want to construct a specific scenario to verify the sorting logic.
        # The logic calculates 'strength' based on distribution across hands.
        # strength += counts[s] * 10 + ranks[s] / counts[s]
        
        # Let's force the deck to be in a specific order (no shuffle)
        # Deck init creates cards: Suit 0 (0-7), Suit 1 (0-7), Suit 2 (0-7), Suit 3 (0-7)
        # If we don't shuffle, deal() gives:
        # Hand 0: Suit 0 (all 8 cards)
        # Hand 1: Suit 1 (all 8 cards)
        # Hand 2: Suit 2 (all 8 cards)
        # Hand 3: Suit 3 (all 8 cards)
        
        # In this case:
        # For Suit 0 (Hand 0): count=8, ranks sum=28. Strength = 80 + 28/8 = 83.5
        # For Suit 1 (Hand 1): same -> 83.5
        # ...
        # All strengths equal.
        # Sorting key: (-strength, s).
        # Since strengths are equal, it sorts by s (original suit index).
        # So 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 3.
        
        d = Deck() # mock_shuffle ensures cards are not shuffled
        
        # Let's manually rearrange cards to create different strengths.
        # We need to manipulate d.cards so that deal() produces specific distributions.
        # deal() takes slices: 0-8, 8-16, 16-24, 24-32.
        
        # Scenario:
        # Suit 0: Distributed 2 cards per player.
        # Suit 1: Concentrated in one player.
        
        # We need to construct the deck list manually.
        cards = []
        
        # Suit 0: 2 cards per player.
        # Ranks 0,1 to P1; 2,3 to P2; 4,5 to P3; 6,7 to P4.
        s0_cards = [Card(0, r) for r in range(8)]
        
        # Suit 1: All to P1.
        s1_cards = [Card(1, r) for r in range(8)]
        
        # Suit 2: All to P2.
        s2_cards = [Card(2, r) for r in range(8)]
        
        # Suit 3: Split P3, P4.
        s3_cards = [Card(3, r) for r in range(8)]
        
        # Construct hands
        # Hand 0 (indices 0-7): Suit 1 (8 cards)
        # Hand 1 (indices 8-15): Suit 2 (8 cards)
        # Hand 2 (indices 16-23): Suit 3 (4 cards), Suit 0 (4 cards)? No, we need to be careful.
        # Total 8 cards per hand.
        
        # Let's try a simpler setup.
        # Hand 0: Suit 0 (8 cards). Strength = 83.5.
        # Hand 1: Suit 1 (4 cards, ranks 0-3), Suit 2 (4 cards, ranks 0-3).
        #         Suit 1: 4*10 + 6/4 = 41.5.
        #         Suit 2: 4*10 + 6/4 = 41.5.
        # Hand 2: Suit 1 (4 cards, ranks 4-7), Suit 2 (4 cards, ranks 4-7).
        #         Suit 1: 4*10 + 22/4 = 40 + 5.5 = 45.5.
        #         Suit 2: 4*10 + 22/4 = 45.5.
        # Hand 3: Suit 3 (8 cards). Strength = 83.5.
        
        # Total Strengths:
        # Suit 0: 83.5
        # Suit 1: 41.5 + 45.5 = 87.0
        # Suit 2: 41.5 + 45.5 = 87.0
        # Suit 3: 83.5
        
        # So Suit 1 and 2 are stronger (87.0) than Suit 0 and 3 (83.5).
        # Tie breaking:
        # Suit 1 vs Suit 2: Strengths equal. Tie break by original index 's'.
        # Suit 1 comes before Suit 2?
        # Sort key is (-strength, s).
        # Suit 1: (-87.0, 1)
        # Suit 2: (-87.0, 2)
        # Suit 0: (-83.5, 0)
        # Suit 3: (-83.5, 3)
        
        # Sorted order:
        # 1. Suit 1
        # 2. Suit 2
        # 3. Suit 0
        # 4. Suit 3
        
        # Mapping:
        # Suit 1 -> 0 (Canonical 0)
        # Suit 2 -> 1 (Canonical 1)
        # Suit 0 -> 2 (Canonical 2)
        # Suit 3 -> 3 (Canonical 3)
        
        # Construct the deck to match this distribution.
        # Hand 0: Suit 0 (all)
        h0 = [Card(0, r) for r in range(8)]
        
        # Hand 1: Suit 1 (0-3), Suit 2 (0-3)
        h1 = [Card(1, r) for r in range(4)] + [Card(2, r) for r in range(4)]
        
        # Hand 2: Suit 1 (4-7), Suit 2 (4-7)
        h2 = [Card(1, r) for r in range(4, 8)] + [Card(2, r) for r in range(4, 8)]
        
        # Hand 3: Suit 3 (all)
        h3 = [Card(3, r) for r in range(8)]
        
        d.cards = h0 + h1 + h2 + h3
        
        transform, reverse = d.transform_suits_canonical()
        
        # Verify mapping
        assert transform(1) == 0
        assert transform(2) == 1
        assert transform(0) == 2
        assert transform(3) == 3
        
        # Verify cards updated
        # First card in deck was Suit 0. Should now be Suit 2.
        assert d.cards[0].suit == 2
