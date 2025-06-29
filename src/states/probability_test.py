import unittest
import numpy as np
import torch
from unittest.mock import patch
import sys
import os

# Assuming the Probability class is in a file called probability.py
# If it's in a different file, adjust the import accordingly
from probability import Probability


class ProbabilityTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.prob = Probability()
    
    def test_init(self):
        """Test initialization of probability matrix."""
        self.assertEqual(self.prob.matrix.shape, (4, 4, 8))
        self.assertEqual(self.prob.matrix.dtype, np.float32)
        self.assertTrue(np.all(np.isclose(self.prob.matrix, 0.25)))
    
    def test_validate_indices(self):
        """Test index validation."""
        # Valid indices should not raise
        self.prob._validate_indices(0, 0, 0)
        self.prob._validate_indices(3, 3, 7)
        
        # Invalid indices should raise ValueError
        with self.assertRaises(ValueError):
            self.prob._validate_indices(-1, 0, 0)
        with self.assertRaises(ValueError):
            self.prob._validate_indices(4, 0, 0)
        with self.assertRaises(ValueError):
            self.prob._validate_indices(0, -1, 0)
        with self.assertRaises(ValueError):
            self.prob._validate_indices(0, 4, 0)
        with self.assertRaises(ValueError):
            self.prob._validate_indices(0, 0, -1)
        with self.assertRaises(ValueError):
            self.prob._validate_indices(0, 0, 8)
    
    def test_update_absolute_value_1(self):
        """Test update with absolute value 1.0."""
        # Initialize with some values
        self.prob.matrix[0, 2, 4] = 0.25
        self.prob.matrix[1, 2, 4] = 0.25
        self.prob.matrix[2, 2, 4] = 0.25
        self.prob.matrix[3, 2, 4] = 0.25
        
        # Update player 1 to have card (2, 4) with probability 1.0
        success = self.prob.update(1, 2, 4, 1.0)
        self.assertTrue(success)
        
        # Check that player 1 has probability 1.0
        self.assertEqual(self.prob.matrix[1, 2, 4], 1.0)
        
        # Check that other players have probability 0.0
        for player in [0, 2, 3]:
            self.assertEqual(self.prob.matrix[player, 2, 4], 0.0)
    
    def test_update_absolute_value_minus_1(self):
        """Test update with absolute value -1.0 (card played)."""
        # Initialize with some values
        self.prob.matrix[0, 2, 4] = 0.25
        self.prob.matrix[1, 2, 4] = 0.25
        self.prob.matrix[2, 2, 4] = 0.25
        self.prob.matrix[3, 2, 4] = 0.25
        
        # Mark card as played
        success = self.prob.update(1, 2, 4, -1.0)
        self.assertTrue(success)
        
        # Check that player 1 has probability -1.0
        self.assertEqual(self.prob.matrix[1, 2, 4], -1.0)
        
        # Check that other players have probability 0.0
        for player in [0, 2, 3]:
            self.assertEqual(self.prob.matrix[player, 2, 4], 0.0)
    
    def test_update_played_card_fails(self):
        """Test that updating a played card fails."""
        # Mark card as played
        self.prob.matrix[1, 2, 4] = -1.0
        
        # Try to update the same card - should fail
        success = self.prob.update(0, 2, 4, 0.5)
        self.assertFalse(success)
    
    def test_update_fractional_probability_with_existing_values(self):
        """Test update with fractional probability values when matrix has existing values."""
        # Initialize with equal probabilities
        self.prob.matrix[0, 2, 4] = 0.25
        self.prob.matrix[1, 2, 4] = 0.25
        self.prob.matrix[2, 2, 4] = 0.25
        self.prob.matrix[3, 2, 4] = 0.25
        
        # Update player 1 to have card (2, 4) with probability 0.6
        success = self.prob.update(1, 2, 4, 0.6)
        self.assertTrue(success)
        
        # Check that the sum of absolute values is approximately 1.0
        total = sum(abs(self.prob.matrix[p, 2, 4]) for p in range(4))
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Check that player 1 has the correct probability
        self.assertAlmostEqual(self.prob.matrix[1, 2, 4], 0.6, places=5)
        
        # Check that other players have proportional remaining probability
        remaining = 0.4
        for player in [0, 2, 3]:
            self.assertAlmostEqual(self.prob.matrix[player, 2, 4], remaining / 3, places=5)
    
    def test_update_fractional_probability_from_zero(self):
        """Test update with fractional probability values starting from zero matrix."""
        # Update player 1 to have card (2, 4) with probability 0.6
        success = self.prob.update(1, 2, 4, 0.6)
        self.assertTrue(success)
        
        # Check that the sum of absolute values is approximately 1.0
        total = sum(abs(self.prob.matrix[p, 2, 4]) for p in range(4))
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Check that player 1 has the correct probability
        self.assertAlmostEqual(self.prob.matrix[1, 2, 4], 0.6, places=5)
        
        # Check that other players have equal remaining probability
        remaining = 0.4
        for player in [0, 2, 3]:
            self.assertAlmostEqual(self.prob.matrix[player, 2, 4], remaining / 3, places=5)
    
    def test_update_no_change(self):
        """Test update when new value equals current value."""
        # Set initial value
        self.prob.matrix[1, 2, 4] = 0.3
        
        # Update with same value
        success = self.prob.update(1, 2, 4, 0.3)
        self.assertTrue(success)
        
        # Value should remain the same
        self.assertAlmostEqual(self.prob.matrix[1, 2, 4], 0.3, places=5)
    
    def test_update_with_negative_probabilities(self):
        """Test update with negative probability values."""
        # Set some negative probabilities (not -1.0)
        self.prob.matrix[0, 1, 2] = -0.2
        self.prob.matrix[1, 1, 2] = 0.4
        self.prob.matrix[2, 1, 2] = 0.5
        self.prob.matrix[3, 1, 2] = 0.3
        
        # Update player 0 to have probability 0.7
        success = self.prob.update(0, 1, 2, 0.7)
        self.assertTrue(success)
        
        # Check that sum of absolute values is approximately 1.0
        total = sum(abs(self.prob.matrix[p, 1, 2]) for p in range(4))
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Check that player 0 has the correct probability
        self.assertAlmostEqual(self.prob.matrix[0, 1, 2], 0.7, places=5)
    
    def test_extract_valid_percentage(self):
        """Test extract method with valid percentage."""
        # Initialize with some values
        self.prob.matrix[0, 2, 4] = 0.1
        self.prob.matrix[1, 2, 4] = 0.2
        self.prob.matrix[2, 2, 4] = 0.3
        self.prob.matrix[3, 2, 4] = 0.4
        
        # Get initial values
        initial_player_value = self.prob.matrix[1, 2, 4]
        initial_others_sum = sum(self.prob.matrix[p, 2, 4] for p in range(4) if p != 1)
        
        # Extract 50% of others' probability
        success = self.prob.extract(1, 2, 4, 0.5)
        self.assertTrue(success)
        
        # Check that player 1's probability increased
        self.assertGreater(self.prob.matrix[1, 2, 4], initial_player_value)
        
        # Check that sum of absolute values is approximately 1.0
        total = sum(abs(self.prob.matrix[p, 2, 4]) for p in range(4))
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Check that extracted amount is correct
        expected_extracted = initial_others_sum * 0.5
        expected_new_value = initial_player_value + expected_extracted
        self.assertAlmostEqual(self.prob.matrix[1, 2, 4], expected_new_value, places=5)
    
    def test_extract_invalid_percentage(self):
        """Test extract method with invalid percentage."""
        # Test negative percentage
        with self.assertRaises(ValueError):
            self.prob.extract(1, 2, 4, -0.1)
        
        # Test percentage > 1
        with self.assertRaises(ValueError):
            self.prob.extract(1, 2, 4, 1.1)
    
    def test_extract_no_others_probability(self):
        """Test extract when other players have no probability."""
        # Set up scenario where only target player has probability
        # Use update() to properly redistribute probabilities
        self.prob.update(1, 2, 4, 1.0)  # This will set others to 0.0
        
        # Try to extract - should fail
        success = self.prob.extract(1, 2, 4, 0.5)
        self.assertFalse(success)
    
    def test_extract_played_card(self):
        """Test extract on played card fails."""
        # Mark card as played
        self.prob.update(1, 2, 4, -1.0)
        
        # Try to extract - should fail
        success = self.prob.extract(0, 2, 4, 0.5)
        self.assertFalse(success)
    
    def test_to_tensor(self):
        """Test conversion to PyTorch tensor."""
        # Set some values
        self.prob.matrix[0, 1, 2] = 0.5
        self.prob.matrix[1, 2, 3] = -1.0
        
        tensor = self.prob.to_tensor()
        
        # Check type and shape
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (4, 4, 8))
        self.assertEqual(tensor.dtype, torch.float32)
        
        # Check that values match
        self.assertTrue(torch.allclose(tensor, torch.tensor(self.prob.matrix)))
    
    def test_transform_matrix(self):
        """Test matrix transformation with steps."""
        # Set some specific values
        self.prob.matrix[0, 1, 2] = 0.5
        self.prob.matrix[1, 2, 3] = 0.8
        self.prob.matrix[2, 0, 7] = -1.0
        
        # Store original matrix
        original_matrix = np.copy(self.prob.matrix)
        
        # Transform with player_step=1
        result = self.prob.rotate(player_step=1)
        
        # Check that method returns self
        self.assertIs(result, self.prob)
        
        # Check that transformation was applied correctly
        for player in range(4):
            for suit in range(4):
                for rank in range(8):
                    original_player = (player - 1) % 4
                    self.assertAlmostEqual(
                        self.prob.matrix[player, suit, rank],
                        original_matrix[original_player, suit, rank],
                        places=5
                    )
    
    def test_transform_matrix_all_steps(self):
        """Test matrix transformation with all step types."""
        # Set some specific values
        self.prob.matrix[0, 1, 2] = 0.5
        self.prob.matrix[1, 2, 3] = 0.8
        self.prob.matrix[2, 0, 7] = -1.0
        
        # Store original matrix
        original_matrix = np.copy(self.prob.matrix)
        
        # Transform with all steps
        self.prob.rotate(player_step=1, suit_step=2, rank_step=3)
        
        # Check that transformation was applied correctly
        for player in range(4):
            for suit in range(4):
                for rank in range(8):
                    original_player = (player - 1) % 4
                    original_suit = (suit - 2) % 4
                    original_rank = (rank - 3) % 8
                    self.assertAlmostEqual(
                        self.prob.matrix[player, suit, rank],
                        original_matrix[original_player, original_suit, original_rank],
                        places=5
                    )
    
    def test_change_suits(self):
        """Test change_suits method."""
        # Set some specific values
        self.prob.matrix[0, 0, 2] = 0.5
        self.prob.matrix[1, 1, 3] = 0.8
        self.prob.matrix[2, 2, 4] = -1.0
        self.prob.matrix[3, 3, 5] = 0.3
        
        # Store original matrix
        original_matrix = np.copy(self.prob.matrix)
        
        # Define a transformation function (swap suits 0<->1, 2<->3)
        def rotate(suit):
            if suit == 0:
                return 1
            elif suit == 1:
                return 0
            elif suit == 2:
                return 3
            else:  # suit == 3
                return 2
        
        result = self.prob.change_suits(rotate)
        
        # Check that method returns self
        self.assertIs(result, self.prob)
        
        # Verify transformation was applied correctly
        for player in range(4):
            for suit in range(4):
                for rank in range(8):
                    original_suit = rotate(suit)  # Inverse transformation for checking
                    expected_value = original_matrix[player, original_suit, rank]
                    self.assertAlmostEqual(
                        self.prob.matrix[player, suit, rank],
                        expected_value,
                        places=5
                    )
    
    def test_copy(self):
        """Test copy method."""
        # Set some values
        self.prob.matrix[0, 1, 2] = 0.5
        self.prob.matrix[1, 2, 3] = -1.0
        
        # Create copy
        prob_copy = self.prob.copy()
        
        # Check that it's a different object
        self.assertIsNot(prob_copy, self.prob)
        self.assertIsNot(prob_copy.matrix, self.prob.matrix)
        
        # Check that values are the same
        self.assertTrue(np.array_equal(prob_copy.matrix, self.prob.matrix))
        
        # Modify original and ensure copy is unchanged
        self.prob.matrix[0, 0, 0] = 0.999
        self.assertNotEqual(prob_copy.matrix[0, 0, 0], 0.999)
    
    def test_getstate_setstate(self):
        """Test serialization methods."""
        # Set some values
        self.prob.matrix[0, 1, 2] = 0.5
        self.prob.matrix[1, 2, 3] = -1.0
        
        # Get state
        state = self.prob.__getstate__()
        
        # Check that state contains matrix as list
        self.assertIn("matrix", state)
        self.assertIsInstance(state["matrix"], list)
        
        # Create new probability object and set state
        new_prob = Probability()
        new_prob.__setstate__(state)
        
        # Check that matrices are equal
        self.assertTrue(np.array_equal(new_prob.matrix, self.prob.matrix))
        self.assertEqual(new_prob.matrix.dtype, np.float32)
    
    def test_probability_conservation_after_multiple_updates(self):
        """Test that probability is conserved across multiple operations."""
        # Initialize with some values
        self.prob.matrix[0, 1, 3] = 0.2
        self.prob.matrix[1, 1, 3] = 0.3
        self.prob.matrix[2, 1, 3] = 0.3
        self.prob.matrix[3, 1, 3] = 0.2
        
        # Perform multiple updates
        self.prob.update(0, 1, 3, 0.5)
        total1 = sum(abs(self.prob.matrix[p, 1, 3]) for p in range(4))
        self.assertAlmostEqual(total1, 1.0, places=5)
        
        self.prob.update(2, 1, 3, 0.1)
        total2 = sum(abs(self.prob.matrix[p, 1, 3]) for p in range(4))
        self.assertAlmostEqual(total2, 1.0, places=5)
        
        # Test extract
        self.prob.extract(1, 1, 3, 0.3)
        total3 = sum(abs(self.prob.matrix[p, 1, 3]) for p in range(4))
        self.assertAlmostEqual(total3, 1.0, places=5)
    
    def test_edge_cases_near_zero(self):
        """Test behavior with very small probability values."""
        # Set very small values
        self.prob.matrix[0, 1, 2] = 1e-8
        self.prob.matrix[1, 1, 2] = 1e-8
        self.prob.matrix[2, 1, 2] = 1e-8
        self.prob.matrix[3, 1, 2] = 1e-8
        
        # Update with a reasonable value
        success = self.prob.update(0, 1, 2, 0.5)
        self.assertTrue(success)
        
        # Check that sum is approximately 1.0
        total = sum(abs(self.prob.matrix[p, 1, 2]) for p in range(4))
        self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_extract_zero_percentage(self):
        """Test extract with 0% - should not change anything."""
        # Initialize with some values
        self.prob.matrix[0, 2, 4] = 0.1
        self.prob.matrix[1, 2, 4] = 0.2
        self.prob.matrix[2, 2, 4] = 0.3
        self.prob.matrix[3, 2, 4] = 0.4
        
        # Store original values
        original_matrix = np.copy(self.prob.matrix)
        
        # Extract 0%
        success = self.prob.extract(1, 2, 4, 0.0)
        self.assertTrue(success)
        
        # Check that nothing changed
        self.assertTrue(np.array_equal(self.prob.matrix, original_matrix))
    
    def test_extract_full_percentage(self):
        """Test extract with 100% - should take all from others."""
        # Initialize with some values
        self.prob.matrix[0, 2, 4] = 0.1
        self.prob.matrix[1, 2, 4] = 0.2
        self.prob.matrix[2, 2, 4] = 0.3
        self.prob.matrix[3, 2, 4] = 0.4
        
        initial_player_value = self.prob.matrix[1, 2, 4]
        initial_others_sum = sum(self.prob.matrix[p, 2, 4] for p in range(4) if p != 1)
        
        # Extract 100%
        success = self.prob.extract(1, 2, 4, 1.0)
        self.assertTrue(success)
        
        # Check that player 1 got everything
        expected_new_value = initial_player_value + initial_others_sum
        self.assertAlmostEqual(self.prob.matrix[1, 2, 4], expected_new_value, places=5)
        
        # Check that sum is still 1.0
        total = sum(abs(self.prob.matrix[p, 2, 4]) for p in range(4))
        self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(ProbabilityTest)
    
    # Run the tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")