import pytest
from card import Card
from trump import Trump, TrumpMode

class TestCard:
    def test_card_initialization(self):
        c = Card(0, 0) # 7 of Spades (assuming 0=Spades, 0=7)
        assert c.suit == 0
        assert c.rank == 0
        
        with pytest.raises(AssertionError):
            Card(4, 0) # Invalid suit
            
        with pytest.raises(AssertionError):
            Card(0, 8) # Invalid rank

    def test_card_value_no_trump(self):
        trump = Trump(TrumpMode.NoTrump, None)
        
        # Ace (rank 7) -> 19
        assert Card(0, 7).value(trump) == 19
        # Ten (rank 3) -> 10
        assert Card(0, 3).value(trump) == 10
        # King (rank 6) -> 4
        assert Card(0, 6).value(trump) == 4
        # Queen (rank 5) -> 3
        assert Card(0, 5).value(trump) == 3
        # Jack (rank 4) -> 2
        assert Card(0, 4).value(trump) == 2
        # 9, 8, 7 -> 0
        assert Card(0, 2).value(trump) == 0
        assert Card(0, 1).value(trump) == 0
        assert Card(0, 0).value(trump) == 0

    def test_card_value_all_trump(self):
        trump = Trump(TrumpMode.AllTrump, None)
        
        # Jack (rank 4) -> 20
        assert Card(0, 4).value(trump) == 20
        # Nine (rank 2) -> 14
        assert Card(0, 2).value(trump) == 14
        # Ace (rank 7) -> 11
        assert Card(0, 7).value(trump) == 11
        # Ten (rank 3) -> 10
        assert Card(0, 3).value(trump) == 10
        # King (rank 6) -> 4
        assert Card(0, 6).value(trump) == 4
        # Queen (rank 5) -> 3
        assert Card(0, 5).value(trump) == 3
        # 8, 7 -> 0
        assert Card(0, 1).value(trump) == 0
        assert Card(0, 0).value(trump) == 0

    def test_card_value_regular_trump_suit(self):
        # Spades is trump (suit 0)
        trump = Trump(TrumpMode.Regular, 0)
        
        # Jack of Spades (Trump) -> 20
        assert Card(0, 4).value(trump) == 20
        # Nine of Spades (Trump) -> 14
        assert Card(0, 2).value(trump) == 14
        # Ace of Spades (Trump) -> 11
        assert Card(0, 7).value(trump) == 11
        
    def test_card_value_regular_non_trump_suit(self):
        # Spades is trump (suit 0)
        trump = Trump(TrumpMode.Regular, 0)
        
        # Testing Hearts (suit 1)
        # Jack of Hearts (Non-Trump) -> 2
        assert Card(1, 4).value(trump) == 2
        # Nine of Hearts (Non-Trump) -> 0
        assert Card(1, 2).value(trump) == 0
        # Ace of Hearts (Non-Trump) -> 11
        assert Card(1, 7).value(trump) == 11

    def test_beats_same_suit(self):
        trump = Trump(TrumpMode.NoTrump, None)
        # Ace beats King
        c1 = Card(0, 7) # Ace
        c2 = Card(0, 6) # King
        assert c1.beats(trump, c2) == True
        assert c2.beats(trump, c1) == False

    def test_beats_different_suit_no_trump(self):
        trump = Trump(TrumpMode.NoTrump, None)
        # Spades vs Hearts
        c1 = Card(0, 7) # Ace Spades
        c2 = Card(1, 7) # Ace Hearts
        
        # If suits are different and no trump involved (or NoTrump mode),
        # the challenger (self) cannot beat the current winner (next)
        # unless self is trump (which is impossible in NoTrump or if self is not trump suit).
        
        # c1 (Spades) vs c2 (Hearts). c1 does NOT beat c2.
        assert c1.beats(trump, c2) == False
        
        # c2 (Hearts) vs c1 (Spades). c2 does NOT beat c1.
        assert c2.beats(trump, c1) == False

    def test_beats_trump(self):
        # Spades is trump
        trump = Trump(TrumpMode.Regular, 0)
        
        c_trump = Card(0, 0) # 7 Spades (Trump)
        c_non_trump = Card(1, 7) # Ace Hearts (Non-Trump)
        
        # Trump should beat non-trump
        assert c_trump.beats(trump, c_non_trump) == True
        
        # Non-trump vs Trump
        # self (non-trump) vs next (trump)
        # self.suit (1) != next.suit (0)
        # returns next.suit (0) != trump.suit (0) -> False.
        # So self does NOT beat next. Correct.
        assert c_non_trump.beats(trump, c_trump) == False

    def test_beats_same_suit_values(self):
        trump = Trump(TrumpMode.Regular, 0) # Spades trump
        
        # Jack Spades (20) vs Ace Spades (11)
        c_jack = Card(0, 4)
        c_ace = Card(0, 7)
        
        assert c_jack.beats(trump, c_ace) == True
        assert c_ace.beats(trump, c_jack) == False
        
    def test_beats_same_suit_ranks(self):
        trump = Trump(TrumpMode.Regular, 0) # Spades trump
        
        # 8 Spades (0) vs 7 Spades (0)
        c_8 = Card(0, 1)
        c_7 = Card(0, 0)
        
        assert c_8.beats(trump, c_7) == True
        assert c_7.beats(trump, c_8) == False

    def test_is_trump(self):
        # Regular mode
        trump_spades = Trump(TrumpMode.Regular, 0) # Spades is trump
        assert Card(0, 7).is_trump(trump_spades) is True  # Spade is trump
        assert Card(1, 7).is_trump(trump_spades) is False # Heart is not trump

        # AllTrump mode
        trump_all = Trump(TrumpMode.AllTrump, None)
        assert Card(0, 7).is_trump(trump_all) is True
        assert Card(1, 7).is_trump(trump_all) is True
        assert Card(2, 7).is_trump(trump_all) is True
        assert Card(3, 7).is_trump(trump_all) is True

        # NoTrump mode
        trump_no = Trump(TrumpMode.NoTrump, None)
        assert Card(0, 7).is_trump(trump_no) is False
        assert Card(1, 7).is_trump(trump_no) is False
