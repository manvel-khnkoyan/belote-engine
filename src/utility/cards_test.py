import pytest
from src.models.card import Card
from src.models.trump import Trump, TrumpMode
from cards import Cards


class TestCards:
    def test_winner_empty(self):
        """Test winner with empty list"""
        trump = Trump(TrumpMode.NoTrump, None)
        card, idx = Cards.winner([], trump)
        assert card is None
        assert idx is None

    def test_winner_single(self):
        """Test winner with single card"""
        c = Card(0, 7)  # Ace Spades
        trump = Trump(TrumpMode.NoTrump, None)
        card, idx = Cards.winner([c], trump)
        assert card == c
        assert idx == 0

    def test_winner_same_suit_higher_rank(self):
        """Test winner when higher rank is played"""
        trump = Trump(TrumpMode.NoTrump, None)
        c1 = Card(0, 7)  # Ace Spades
        c2 = Card(0, 6)  # King Spades
        
        card, idx = Cards.winner([c2, c1], trump)
        assert card == c1  # Ace wins
        assert idx == 1

    def test_winner_same_suit_lower_rank(self):
        """Test winner when led card holds"""
        trump = Trump(TrumpMode.NoTrump, None)
        c1 = Card(0, 7)  # Ace Spades
        c2 = Card(0, 6)  # King Spades
        
        card, idx = Cards.winner([c1, c2], trump)
        assert card == c1  # Ace holds
        assert idx == 0

    def test_winner_different_suit_no_trump(self):
        """Test winner with different suits in no trump mode"""
        trump = Trump(TrumpMode.NoTrump, None)
        c1 = Card(0, 7)  # Ace Spades
        c2 = Card(1, 7)  # Ace Hearts
        
        card, idx = Cards.winner([c1, c2], trump)
        assert card == c1  # Led suit wins
        assert idx == 0

    def test_winner_trump_cut(self):
        """Test winner when trump cuts non-trump"""
        trump = Trump(TrumpMode.Regular, 0)  # Spades trump
        c1 = Card(1, 7)  # Ace Hearts
        c2 = Card(0, 0)  # 7 Spades (trump)
        
        card, idx = Cards.winner([c1, c2], trump)
        assert card == c2  # Trump wins
        assert idx == 1

    def test_winner_over_trump(self):
        """Test winner when higher trump beats lower trump"""
        trump = Trump(TrumpMode.Regular, 0)  # Spades trump
        c1 = Card(0, 2)  # 9 Spades (14)
        c2 = Card(0, 4)  # Jack Spades (20)
        
        card, idx = Cards.winner([c1, c2], trump)
        assert card == c2  # Jack wins
        assert idx == 1

    def test_value_empty(self):
        """Test value of empty list"""
        trump = Trump(TrumpMode.NoTrump, None)
        assert Cards.value([], trump) == 0

    def test_value_single(self):
        """Test value of single card"""
        trump = Trump(TrumpMode.NoTrump, None)
        c = Card(0, 7)  # Ace = 19 in NoTrump
        assert Cards.value([c], trump) == 19

    def test_value_multiple(self):
        """Test value of multiple cards"""
        trump = Trump(TrumpMode.NoTrump, None)
        c1 = Card(0, 7)  # Ace = 19 in NoTrump
        c2 = Card(0, 3)  # 10 = 10
        c3 = Card(0, 6)  # King = 4
        assert Cards.value([c1, c2, c3], trump) == 33

    def test_value_with_trump(self):
        """Test value with trump cards"""
        trump = Trump(TrumpMode.Regular, 0)  # Spades trump
        c1 = Card(0, 4)  # Jack Spades (trump) = 20
        c2 = Card(0, 2)  # 9 Spades (trump) = 14
        c3 = Card(1, 7)  # Ace Hearts = 11
        assert Cards.value([c1, c2, c3], trump) == 45

    def test_sort_regular_trump(self):
        """Test sorting with trump"""
        trump = Trump(TrumpMode.Regular, 0)  # Spades trump
        
        c1 = Card(1, 7)  # Ace Hearts
        c2 = Card(0, 4)  # Jack Spades (trump)
        c3 = Card(0, 2)  # 9 Spades (trump)
        c4 = Card(1, 6)  # King Hearts
        
        cards = [c1, c2, c3, c4]
        Cards.sort(cards, trump)
        
        # Trump first (by value), then non-trump
        assert cards[0] == c2  # Jack Spades
        assert cards[1] == c3  # 9 Spades
        assert cards[2] == c1  # Ace Hearts
        assert cards[3] == c4  # King Hearts

    def test_sort_no_trump(self):
        """Test sorting without trump"""
        trump = Trump(TrumpMode.NoTrump, None)
        
        c1 = Card(0, 7)  # Ace Spades
        c2 = Card(0, 3)  # 10 Spades
        c3 = Card(1, 7)  # Ace Hearts
        
        cards = [c2, c1, c3]
        Cards.sort(cards, trump)
        
        # Spades has higher sum than Hearts
        assert cards[0] == c1  # Ace Spades
        assert cards[1] == c2  # 10 Spades
        assert cards[2] == c3  # Ace Hearts

    def test_sort_by_max_value(self):
        """Test sorting by max card value"""
        trump = Trump(TrumpMode.Regular, 2)  # Diamonds trump
        
        c1 = Card(0, 7)  # Ace Spades (max=11)
        c2 = Card(1, 3)  # 10 Hearts (max=10)
        c3 = Card(1, 1)  # 8 Hearts (value=0)
        
        cards = [c2, c3, c1]
        Cards.sort(cards, trump)
        
        # Spades first (higher max), then Hearts
        assert cards[0] == c1  # Ace Spades
        assert cards[1] == c2  # 10 Hearts
        assert cards[2] == c3  # 8 Hearts

    def test_sort_by_sum_when_max_equal(self):
        """Test sorting by sum when max values equal"""
        trump = Trump(TrumpMode.Regular, 2)  # Diamonds trump
        
        c1 = Card(0, 7)  # Ace Spades (max=11, sum=11)
        c2 = Card(1, 7)  # Ace Hearts (max=11, sum=21)
        c3 = Card(1, 3)  # 10 Hearts
        
        cards = [c2, c3, c1]
        Cards.sort(cards, trump)
        
        # Hearts first (same max, higher sum)
        assert cards[0] == c2  # Ace Hearts
        assert cards[1] == c3  # 10 Hearts
        assert cards[2] == c1  # Ace Spades

    def test_sort_by_count_when_max_and_sum_equal(self):
        """Test sorting by count when max and sum equal"""
        trump = Trump(TrumpMode.Regular, 2)  # Diamonds trump
        
        c1 = Card(0, 7)  # Ace Spades (count=1)
        c2 = Card(1, 7)  # Ace Hearts (count=2)
        c3 = Card(3, 7)  # Ace Clubs (count=1)
        c4 = Card(1, 1)  # 8 Hearts
        
        cards = [c2, c4, c3, c1]
        Cards.sort(cards, trump)
        
        # Hearts first (higher count), then by suit number
        assert cards[0] == c2  # Ace Hearts
        assert cards[1] == c4  # 8 Hearts
        assert cards[2] == c1  # Ace Spades (suit 0 < 3)
        assert cards[3] == c3  # Ace Clubs
