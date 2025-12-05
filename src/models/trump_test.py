import pytest
from trump import Trump, TrumpMode
from suits import Suits

class TestTrump:
    def test_trump_initialization_regular(self):
        # Regular mode requires a suit
        t = Trump(TrumpMode.Regular, 0)
        assert t.mode == TrumpMode.Regular
        assert t.suit == 0

    def test_trump_initialization_no_trump(self):
        # NoTrump mode usually has suit as None, but the assert allows 0-3 or None.
        # The code says: assert suit is None or 0 <= suit <= 3
        t = Trump(TrumpMode.NoTrump, None)
        assert t.mode == TrumpMode.NoTrump
        assert t.suit is None

    def test_trump_initialization_all_trump(self):
        t = Trump(TrumpMode.AllTrump, None)
        assert t.mode == TrumpMode.AllTrump
        assert t.suit is None

    def test_trump_initialization_invalid_mode(self):
        with pytest.raises(AssertionError):
            Trump(99, 0)

    def test_trump_initialization_invalid_suit(self):
        with pytest.raises(AssertionError):
            Trump(TrumpMode.Regular, 4)
        
        with pytest.raises(AssertionError):
            Trump(TrumpMode.Regular, -1)

    def test_trump_iter(self):
        t = Trump(TrumpMode.Regular, 2)
        mode, suit = t
        assert mode == TrumpMode.Regular
        assert suit == 2

    def test_trump_repr(self):
        t_no = Trump(TrumpMode.NoTrump, None)
        assert repr(t_no) == "No Trump"

        t_all = Trump(TrumpMode.AllTrump, None)
        assert repr(t_all) == "All Trump"

        # Suits = ['♠', '♥', '♦', '♣']
        t_reg = Trump(TrumpMode.Regular, 0) # Spades
        assert repr(t_reg) == f"Trump {Suits[0]}"
        
        t_reg2 = Trump(TrumpMode.Regular, 1) # Hearts
        assert repr(t_reg2) == f"Trump {Suits[1]}"
