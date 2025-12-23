import pytest
from canonical import Canonical
from src.models.card import Card
from src.models.trump import Trump, TrumpMode

class TestCanonical:
    
    def test_empty_cards_returns_all_suits_mapped(self):
        """Test that empty card list still creates mapping for all suits"""
        cards = []
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, reverse_map = Canonical.create_transform_map(cards, trump)
        
        # Should have all 4 suits
        assert len(can_map) == 4
        assert set(can_map.keys()) == {0, 1, 2, 3}
        assert set(can_map.values()) == {0, 1, 2, 3}
        # Reverse map should be the inverse
        assert len(reverse_map) == 4
        assert all(reverse_map[can_map[orig]] == orig for orig in range(4))
    
    def test_single_suit_highest_strength(self):
        """Test that suit with cards gets highest canonical position"""
        # Only spades (suit 0) with high ranks
        cards = [Card(0, 7), Card(0, 6)]  # Ace, King
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Spades should map to 0 (highest)
        assert can_map[0] == 0
    
    def test_all_suits_equal_strength(self):
        """Test mapping when all suits have equal average strength"""
        # Each suit has one Ace (rank 7)
        cards = [Card(0, 7), Card(1, 7), Card(2, 7), Card(3, 7)]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # All suits equal, so ordered by suit number (0, 1, 2, 3)
        assert can_map[0] == 0
        assert can_map[1] == 1
        assert can_map[2] == 2
        assert can_map[3] == 3
    
    def test_clear_strength_ordering(self):
        """Test ordering with clear strength differences"""
        # Spades: Ace (rank 7)
        # Hearts: King (rank 6)
        # Diamonds: Queen (rank 5)
        # Clubs: Jack (rank 4)
        cards = [
            Card(0, 7),  # Ace Spades
            Card(1, 6),  # King Hearts
            Card(2, 5),  # Queen Diamonds
            Card(3, 4),  # Jack Clubs
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Spades strongest -> 0
        # Hearts second -> 1
        # Diamonds third -> 2
        # Clubs weakest -> 3
        assert can_map[0] == 0
        assert can_map[1] == 1
        assert can_map[2] == 2
        assert can_map[3] == 3
    
    def test_reverse_strength_ordering(self):
        """Test with reverse strength order"""
        # Clubs: Ace (rank 7)
        # Diamonds: King (rank 6)
        # Hearts: Queen (rank 5)
        # Spades: Jack (rank 4)
        cards = [
            Card(3, 7),  # Ace Clubs
            Card(2, 6),  # King Diamonds
            Card(1, 5),  # Queen Hearts
            Card(0, 4),  # Jack Spades
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Clubs strongest -> 0
        # Diamonds second -> 1
        # Hearts third -> 2
        # Spades weakest -> 3
        assert can_map[3] == 0
        assert can_map[2] == 1
        assert can_map[1] == 2
        assert can_map[0] == 3
    
    def test_multiple_cards_average_strength(self):
        """Test strength calculation with multiple cards per suit"""
        # Spades: 7 (rank 0) and Ace (rank 7) -> avg = 3.5
        # Hearts: 10 (rank 3) and 10 (rank 3) -> avg = 3.0
        cards = [
            Card(0, 0),  # 7 Spades
            Card(0, 7),  # Ace Spades
            Card(1, 3),  # 10 Hearts
            Card(1, 3),  # 10 Hearts
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Spades has higher average (3.5 > 3.0)
        assert can_map[0] == 0  # Spades -> 0
        assert can_map[1] == 1  # Hearts -> 1
    
    def test_suit_with_more_cards_but_lower_average(self):
        """Test that average rank matters, not count"""
        # Spades: Ace (rank 7) -> avg = 7.0
        # Hearts: King (rank 6), Queen (rank 5), Jack (rank 4) -> avg = 5.0
        cards = [
            Card(0, 7),  # Ace Spades
            Card(1, 6),  # King Hearts
            Card(1, 5),  # Queen Hearts
            Card(1, 4),  # Jack Hearts
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Spades has higher average despite fewer cards
        assert can_map[0] == 0  # Spades (avg 7) -> 0
        assert can_map[1] == 1  # Hearts (avg 5) -> 1
    
    def test_tie_breaking_by_suit_number(self):
        """Test that ties are broken by original suit number"""
        # Spades: 10 (rank 3) -> avg = 3.0
        # Diamonds: 10 (rank 3) -> avg = 3.0
        cards = [
            Card(0, 3),  # 10 Spades
            Card(2, 3),  # 10 Diamonds
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Same strength, so suit 0 comes before suit 2
        assert can_map[0] == 0  # Spades -> 0
        assert can_map[2] == 1  # Diamonds -> 1
    
    def test_all_suits_present_different_counts(self):
        """Test with all suits having different card counts"""
        cards = [
            # Spades: 3 cards, ranks 7,6,5 -> avg = 6.0
            Card(0, 7), Card(0, 6), Card(0, 5),
            # Hearts: 2 cards, ranks 7,5 -> avg = 6.0
            Card(1, 7), Card(1, 5),
            # Diamonds: 2 cards, ranks 4,3 -> avg = 3.5
            Card(2, 4), Card(2, 3),
            # Clubs: 1 card, rank 2 -> avg = 2.0
            Card(3, 2),
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Spades and Hearts both avg 6.0, Spades comes first (suit 0 < 1)
        # Diamonds avg 3.5
        # Clubs avg 2.0
        assert can_map[0] == 0  # Spades
        assert can_map[1] == 1  # Hearts
        assert can_map[2] == 2  # Diamonds
        assert can_map[3] == 3  # Clubs
    
    def test_partial_suits_only_two_suits(self):
        """Test when only 2 suits have cards"""
        cards = [
            Card(1, 7),  # Ace Hearts -> avg = 7.0
            Card(3, 3),  # 10 Clubs -> avg = 3.0
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Hearts strongest -> 0
        # Clubs second -> 1
        # Spades and Diamonds have no cards (avg 0), order by suit number
        assert can_map[1] == 0  # Hearts
        assert can_map[3] == 1  # Clubs
        assert can_map[0] == 2  # Spades (no cards)
        assert can_map[2] == 3  # Diamonds (no cards)
    
    def test_one_suit_dominates(self):
        """Test when one suit has significantly higher strength"""
        cards = [
            # Hearts: all high cards
            Card(1, 7), Card(1, 6), Card(1, 5), Card(1, 4),  # avg = 5.5
            # Other suits: low cards
            Card(0, 0),  # 7 Spades -> avg = 0
            Card(2, 1),  # 8 Diamonds -> avg = 1.0
            Card(3, 2),  # 9 Clubs -> avg = 2.0
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        assert can_map[1] == 0  # Hearts (strongest)
        assert can_map[3] == 1  # Clubs
        assert can_map[2] == 2  # Diamonds
        assert can_map[0] == 3  # Spades (weakest)
    
    def test_mapping_is_bijective(self):
        """Test that mapping is one-to-one (bijective)"""
        cards = [
            Card(0, 7), Card(1, 6), Card(2, 5), Card(3, 4)
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Each original suit maps to unique canonical suit
        values = list(can_map.values())
        assert len(values) == len(set(values))
        assert set(values) == {0, 1, 2, 3}
    
    def test_all_ranks_in_one_suit(self):
        """Test when one suit has all possible ranks"""
        cards = [Card(2, rank) for rank in range(8)]  # All ranks in Diamonds
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Diamonds should be strongest
        # avg rank = (0+1+2+3+4+5+6+7)/8 = 3.5
        assert can_map[2] == 0
    
    def test_realistic_hand_scenario(self):
        """Test with a realistic 8-card hand"""
        cards = [
            Card(0, 7),  # Ace Spades
            Card(0, 3),  # 10 Spades
            Card(1, 6),  # King Hearts
            Card(1, 2),  # 9 Hearts
            Card(2, 5),  # Queen Diamonds
            Card(2, 1),  # 8 Diamonds
            Card(3, 4),  # Jack Clubs
            Card(3, 0),  # 7 Clubs
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Calculate expected order:
        # Spades: (7+3)/2 = 5.0
        # Hearts: (6+2)/2 = 4.0
        # Diamonds: (5+1)/2 = 3.0
        # Clubs: (4+0)/2 = 2.0
        
        assert can_map[0] == 0  # Spades strongest
        assert can_map[1] == 1  # Hearts
        assert can_map[2] == 2  # Diamonds
        assert can_map[3] == 3  # Clubs weakest
    
    def test_full_deck_all_equal(self):
        """Test with full deck where all suits should be equal"""
        cards = []
        for suit in range(4):
            for rank in range(8):
                cards.append(Card(suit, rank))
        
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # All suits have same average (3.5), so ordered by suit number
        assert can_map[0] == 0
        assert can_map[1] == 1
        assert can_map[2] == 2
        assert can_map[3] == 3
    
    def test_regular_trump_priority(self):
        """Test that regular trump suit is placed first"""
        cards = [
            Card(0, 7),  # Ace Spades (strongest)
            Card(1, 6),  # King Hearts
            Card(2, 5),  # Queen Diamonds
            Card(3, 4),  # Jack Clubs (weakest)
        ]
        # Hearts (suit 1) is trump
        trump = Trump(TrumpMode.Regular, 1)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Hearts must be at position 0 (trump priority)
        assert can_map[1] == 0
        # Spades is strongest among remaining, so position 1
        assert can_map[0] == 1
        # Diamonds position 2
        assert can_map[2] == 2
        # Clubs position 3
        assert can_map[3] == 3
    
    def test_no_trump_normal_ordering(self):
        """Test that NoTrump uses normal strength ordering"""
        cards = [
            Card(0, 7),  # Ace Spades
            Card(1, 6),  # King Hearts
            Card(2, 5),  # Queen Diamonds
            Card(3, 4),  # Jack Clubs
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Normal ordering by strength
        assert can_map[0] == 0  # Spades
        assert can_map[1] == 1  # Hearts
        assert can_map[2] == 2  # Diamonds
        assert can_map[3] == 3  # Clubs
    
    def test_reverse_map_is_valid_inverse(self):
        """Test that reverse map correctly inverts the canonical map"""
        cards = [
            Card(0, 7), Card(1, 6), Card(2, 5), Card(3, 4)
        ]
        trump = Trump(TrumpMode.NoTrump, None)
        can_map, reverse_map = Canonical.create_transform_map(cards, trump)
        
        # reverse_map should be the inverse
        for orig, canon in can_map.items():
            assert reverse_map[canon] == orig
        
        # And vice versa
        for canon, orig in reverse_map.items():
            assert can_map[orig] == canon
    
    def test_trump_spades_with_weak_strength(self):
        """Test trump priority overrides strength ordering"""
        cards = [
            Card(0, 0),  # 7 Spades (weakest)
            Card(1, 7),  # Ace Hearts (strongest)
            Card(2, 6),  # King Diamonds
            Card(3, 5),  # Queen Clubs
        ]
        # Spades is trump despite being weakest
        trump = Trump(TrumpMode.Regular, 0)
        can_map, _ = Canonical.create_transform_map(cards, trump)
        
        # Spades must be at position 0 (trump priority)
        assert can_map[0] == 0
        # Hearts strongest among remaining
        assert can_map[1] == 1
        # Diamonds second
        assert can_map[2] == 2
        # Clubs weakest
        assert can_map[3] == 3
