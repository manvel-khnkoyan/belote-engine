import pytest
import numpy as np
from src.models.history import History

class TestHistory:
    def test_initialization(self):
        hist = History()
        assert hist.matrix.shape == (4, 4, 8)
        np.testing.assert_allclose(hist.matrix, 0.0)

    def test_update(self):
        hist = History()
        # Player 0 plays Suit 0, Rank 0
        assert hist.played(0, 0, 0) == True
        
        assert hist.matrix[0, 0, 0] == pytest.approx(1.0)
        assert hist.matrix[1, 0, 0] == pytest.approx(0.0)
        
        # Check helpers
        # assert hist.is_played(0, 0) == True
        # assert hist.who_played(0, 0) == 0

    def test_update_duplicate(self):
        hist = History()
        hist.played(0, 0, 0)
        
        # Try to play same card again by same player
        assert hist.played(0, 0, 0) == False
        
        # Try to play same card by different player
        assert hist.played(1, 0, 0) == False

    def test_multiple_cards(self):
        hist = History()
        hist.played(0, 0, 0)
        hist.played(1, 1, 1)
        
        # assert hist.who_played(0, 0) == 0
        # assert hist.who_played(1, 1) == 1
        # assert hist.who_played(2, 2) == -1
