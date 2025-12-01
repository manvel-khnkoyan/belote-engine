import pytest
from card import Card
from collection import Collection
from trump import Trump, TrumpMode

class TestList:
    def test_list_initialization(self):
        cards = [Card(0, 0), Card(1, 1)]
        l = Collection(cards)
        assert len(l) == 2
        assert l[0] == cards[0]
        assert l[1] == cards[1]
        
        l2 = Collection()
        assert len(l2) == 0

    def test_winner_empty(self):
        l = Collection()
        trump = Trump(TrumpMode.NoTrump, None)
        card, idx = l.winner(trump)
        assert card is None
        assert idx is None

    def test_winner_single(self):
        c = Card(0, 7)
        l = Collection([c])
        trump = Trump(TrumpMode.NoTrump, None)
        card, idx = l.winner(trump)
        assert card == c
        assert idx == 0

    def test_winner_same_suit_higher_rank(self):
        trump = Trump(TrumpMode.NoTrump, None)
        # Ace Spades, King Spades
        c1 = Card(0, 7) # Ace
        c2 = Card(0, 6) # King
        l = Collection([c2, c1]) # King led, Ace played
        
        # Ace beats King
        card, idx = l.winner(trump)
        assert card == c1
        assert idx == 1

    def test_winner_same_suit_lower_rank(self):
        trump = Trump(TrumpMode.NoTrump, None)
        # Ace Spades, King Spades
        c1 = Card(0, 7) # Ace
        c2 = Card(0, 6) # King
        l = Collection([c1, c2]) # Ace led, King played
        
        # Ace holds
        card, idx = l.winner(trump)
        assert card == c1
        assert idx == 0

    def test_winner_different_suit_no_trump_mode(self):
        trump = Trump(TrumpMode.NoTrump, None)
        # Spades led, Hearts played
        c1 = Card(0, 7) # Ace Spades
        c2 = Card(1, 7) # Ace Hearts
        l = Collection([c1, c2])
        
        # Spades should win (Hearts is discard)
        card, idx = l.winner(trump)
        assert card == c1
        assert idx == 0

    def test_winner_different_suit_regular_mode_no_trump_card(self):
        # Diamonds is trump
        trump = Trump(TrumpMode.Regular, 2)
        
        # Spades led, Hearts played
        c1 = Card(0, 7) # Ace Spades
        c2 = Card(1, 7) # Ace Hearts
        l = Collection([c1, c2])
        
        # Spades should win
        card, idx = l.winner(trump)
        assert card == c1
        assert idx == 0

    def test_winner_trump_cut(self):
        # Spades is trump
        trump = Trump(TrumpMode.Regular, 0)
        
        # Hearts led, Spades (Trump) played
        c1 = Card(1, 7) # Ace Hearts
        c2 = Card(0, 0) # 7 Spades (Trump)
        l = Collection([c1, c2])
        
        # Trump should win
        card, idx = l.winner(trump)
        assert card == c2
        assert idx == 1

    def test_winner_over_trump(self):
        # Spades is trump
        trump = Trump(TrumpMode.Regular, 0)
        
        # Jack Spades (20) vs 9 Spades (14)
        c1 = Card(0, 2) # 9 Spades
        c2 = Card(0, 4) # Jack Spades
        l = Collection([c1, c2])
        
        # Jack beats 9
        card, idx = l.winner(trump)
        assert card == c2
        assert idx == 1

    def test_sort_regular(self):
        # Spades Trump
        trump = Trump(TrumpMode.Regular, 0)
        
        c1 = Card(1, 7) # Ace Hearts (11 in non-trump)
        c2 = Card(0, 4) # Jack Spades (Trump, 20)
        c3 = Card(0, 2) # 9 Spades (Trump, 14)
        c4 = Card(1, 6) # King Hearts (4 in non-trump)
        
        l = Collection([c1, c2, c3, c4])
        l.sort(trump)
        
        # Expected order:
        # Trumps first (by value): Jack Spades (20), 9 Spades (14)
        # Then non-trump suits sorted by max value, sum, count
        # Hearts: max=11, sum=15, count=2
        # Order: Ace Hearts (11), King Hearts (4)
        
        assert l[0] == c2  # Jack Spades
        assert l[1] == c3  # 9 Spades
        assert l[2] == c1  # Ace Hearts
        assert l[3] == c4  # King Hearts

    def test_sort_no_trump(self):
        trump = Trump(TrumpMode.NoTrump, None)
        
        c1 = Card(0, 7) # Ace Spades (11)
        c2 = Card(0, 3) # 10 Spades (10)
        c3 = Card(1, 7) # Ace Hearts (11)
        
        l = Collection([c2, c1, c3])
        l.sort(trump)
        
        # No trump, so suits sorted by max value, sum, count
        # Spades: max=11, sum=21, count=2
        # Hearts: max=11, sum=11, count=1
        # Spades comes first (higher sum)
        # Within Spades: Ace (11) > 10 (10)
        
        assert l[0] == c1  # Ace Spades
        assert l[1] == c2  # 10 Spades
        assert l[2] == c3  # Ace Hearts

    def test_sort_by_max_value(self):
        # Test sorting by max value
        trump = Trump(TrumpMode.Regular, 2)  # Diamonds trump
        
        c1 = Card(0, 7) # Ace Spades (11)
        c2 = Card(1, 3) # 10 Hearts (10)
        c3 = Card(1, 1) # 8 Hearts (0)
        
        l = Collection([c2, c3, c1])
        l.sort(trump)
        
        # Spades: max=11, sum=11, count=1
        # Hearts: max=10, sum=10, count=2
        # Spades comes first (higher max value)
        
        assert l[0] == c1  # Ace Spades
        assert l[1] == c2  # 10 Hearts
        assert l[2] == c3  # 8 Hearts

    def test_sort_by_sum_when_max_equal(self):
        # Test sorting by sum when max values are equal
        trump = Trump(TrumpMode.Regular, 2)  # Diamonds trump
        
        c1 = Card(0, 7) # Ace Spades (11)
        c2 = Card(1, 7) # Ace Hearts (11)
        c3 = Card(1, 3) # 10 Hearts (10)
        
        l = Collection([c2, c3, c1])
        l.sort(trump)
        
        # Spades: max=11, sum=11, count=1
        # Hearts: max=11, sum=21, count=2
        # Hearts comes first (same max, higher sum)
        
        assert l[0] == c2  # Ace Hearts
        assert l[1] == c3  # 10 Hearts
        assert l[2] == c1  # Ace Spades

    def test_sort_by_count_when_max_and_sum_equal(self):
        # Test sorting by count when max and sum are equal
        trump = Trump(TrumpMode.Regular, 2)  # Diamonds trump
        
        c1 = Card(0, 7) # Ace Spades (11)
        c2 = Card(1, 7) # Ace Hearts (11)
        c3 = Card(3, 7) # Ace Clubs (11)
        c4 = Card(1, 1) # 8 Hearts (0)
        
        l = Collection([c2, c4, c3, c1])
        l.sort(trump)
        
        # Spades: max=11, sum=11, count=1
        # Hearts: max=11, sum=11, count=2
        # Clubs: max=11, sum=11, count=1
        # Hearts comes first (same max, same sum, higher count)
        # Then Spades (suit 0) before Clubs (suit 3)
        
        assert l[0] == c2  # Ace Hearts
        assert l[1] == c4  # 8 Hearts
        assert l[2] == c1  # Ace Spades (suit 0 < suit 3)
        assert l[3] == c3  # Ace Clubs
