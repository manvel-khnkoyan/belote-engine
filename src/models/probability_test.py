import pytest
import numpy as np
from probability import Probability

class TestProbability:
    def test_initialization(self):
        prob = Probability()
        # 4 players, 4 suits, 8 ranks
        assert prob.matrix.shape == (4, 4, 8)
        # All initialized to 0.25
        np.testing.assert_allclose(prob.matrix, 0.25)

    def test_update_certainty_has_card(self):
        prob = Probability()
        # Player 0 has Suit 0, Rank 0
        prob.update(0, 0, 0, 1.0)
        
        assert prob.matrix[0, 0, 0] == 1.0
        assert prob.matrix[1, 0, 0] == 0.0
        assert prob.matrix[2, 0, 0] == 0.0
        assert prob.matrix[3, 0, 0] == 0.0

    def test_update_certainty_played_card(self):
        prob = Probability()
        # Player 0 played Suit 0, Rank 0
        prob.update(0, 0, 0, -1.0)
        
        # Now we expect 0.0 for everyone, meaning the card is out of play
        assert prob.matrix[0, 0, 0] == 0.0
        assert prob.matrix[1, 0, 0] == 0.0
        assert prob.matrix[2, 0, 0] == 0.0
        assert prob.matrix[3, 0, 0] == 0.0
        
        # Try to update again (should return False because sum is 0)
        assert prob.update(0, 0, 0, 0.5) == False

    def test_update_probability_redistribution(self):
        prob = Probability()
        # Initial: [0.25, 0.25, 0.25, 0.25]
        # Update P0 to 0.4
        prob.update(0, 0, 0, 0.4)
        
        assert prob.matrix[0, 0, 0] == pytest.approx(0.4)
        # Others should be 0.2
        # Remaining 0.6 distributed among 3 players (0.75 total previously)
        # 0.25 * (0.6 / 0.75) = 0.2
        np.testing.assert_allclose(prob.matrix[1:, 0, 0], 0.2)
        
        assert abs(np.sum(prob.matrix[:, 0, 0]) - 1.0) < 1e-6

    def test_update_probability_uneven(self):
        prob = Probability()
        # Set up uneven state: P0=0.4, others=0.2
        prob.update(0, 0, 0, 0.4)
        
        # Now update P0 to 0.7
        prob.update(0, 0, 0, 0.7)
        
        assert prob.matrix[0, 0, 0] == pytest.approx(0.7)
        # Remaining 0.3. Others sum was 0.6.
        # Factor = 0.3 / 0.6 = 0.5
        # Others should be 0.2 * 0.5 = 0.1
        np.testing.assert_allclose(prob.matrix[1:, 0, 0], 0.1)
        
        assert abs(np.sum(prob.matrix[:, 0, 0]) - 1.0) < 1e-6

    def test_update_from_certainty_to_uncertainty(self):
        prob = Probability()
        # P0 has 1.0. Others 0.0.
        prob.update(0, 0, 0, 1.0)
        
        # Update P0 to 0.4.
        prob.update(0, 0, 0, 0.4)
        
        assert prob.matrix[0, 0, 0] == pytest.approx(0.4)
        # Remaining 0.6. Others sum 0.0.
        # Others get 0.6 / 3 = 0.2.
        np.testing.assert_allclose(prob.matrix[1:, 0, 0], 0.2)

    def test_extract(self):
        prob = Probability()
        # Initial: [0.25, 0.25, 0.25, 0.25]
        # Extract 50% from others to P0
        # Others sum = 0.75. 50% of that is 0.375.
        # P0 new = 0.25 + 0.375 = 0.625
        
        prob.extract(0, 0, 0, 0.5)
        
        assert prob.matrix[0, 0, 0] == 0.625
        # Remaining 0.375 distributed among others (who had 0.75)
        # Factor = 0.375 / 0.75 = 0.5
        # Others new = 0.25 * 0.5 = 0.125
        np.testing.assert_allclose(prob.matrix[1:, 0, 0], 0.125)
        
        assert abs(np.sum(prob.matrix[:, 0, 0]) - 1.0) < 1e-6

    def test_extract_played_card(self):
        prob = Probability()
        prob.update(0, 0, 0, -1.0)
        assert prob.extract(0, 0, 0, 0.5) == False

    def test_extract_no_others(self):
        prob = Probability()
        prob.update(0, 0, 0, 1.0) # P0 has it, others 0
        # Extract from others (sum=0)
        assert prob.extract(0, 0, 0, 0.5) == False
