import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Assuming the classes are in their respective files
from src.states.probability import Probability
from suits_canonical_transformer import SuitsCanonicalTransformer


class SuitsCanonicalTransformerTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.prob = Probability()
        
        # Create a test scenario with different suit strengths
        # Suit 0: weak (low probabilities)
        # Suit 1: medium strength
        # Suit 2: strong (high probabilities)
        # Suit 3: medium-weak
        
        # Player 0 probabilities
        self.prob.matrix[0, 0, :] = [0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0]  # Suit 0: weak
        self.prob.matrix[0, 1, :] = [0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2]  # Suit 1: medium
        self.prob.matrix[0, 2, :] = [0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7]  # Suit 2: strong
        self.prob.matrix[0, 3, :] = [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1]  # Suit 3: medium-weak
        
        # Player 1 probabilities (similar pattern)
        self.prob.matrix[1, 0, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        self.prob.matrix[1, 1, :] = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        self.prob.matrix[1, 2, :] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        self.prob.matrix[1, 3, :] = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        
        # Players 2 and 3 with similar but varied patterns
        self.prob.matrix[2, 0, :] = [0.08, 0.02, 0.08, 0.02, 0.08, 0.02, 0.08, 0.02]
        self.prob.matrix[2, 1, :] = [0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3]
        self.prob.matrix[2, 2, :] = [0.85, 0.8, 0.85, 0.8, 0.85, 0.8, 0.85, 0.8]
        self.prob.matrix[2, 3, :] = [0.18, 0.12, 0.18, 0.12, 0.18, 0.12, 0.18, 0.12]
        
        self.prob.matrix[3, 0, :] = [0.06, 0.04, 0.06, 0.04, 0.06, 0.04, 0.06, 0.04]
        self.prob.matrix[3, 1, :] = [0.35, 0.28, 0.35, 0.28, 0.35, 0.28, 0.35, 0.28]
        self.prob.matrix[3, 2, :] = [0.95, 0.88, 0.95, 0.88, 0.95, 0.88, 0.95, 0.88]
        self.prob.matrix[3, 3, :] = [0.22, 0.14, 0.22, 0.14, 0.22, 0.14, 0.22, 0.14]
    
    def test_init_valid_probability(self):
        """Test initialization with valid probability object."""
        transformer = SuitsCanonicalTransformer(self.prob)
        
        self.assertEqual(transformer.num_players, 4)
        self.assertEqual(transformer.num_suits, 4)
        self.assertEqual(transformer.num_ranks, 8)
        self.assertIsNotNone(transformer._suit_map)
        self.assertIsNotNone(transformer._reverse_map)
    
    def test_init_none_probability(self):
        """Test initialization with None probability raises ValueError."""
        with self.assertRaises(ValueError) as context:
            SuitsCanonicalTransformer(None)
        self.assertIn("Invalid probability matrix", str(context.exception))
    
    def test_init_probability_none_matrix(self):
        """Test initialization with probability having None matrix raises ValueError."""
        prob = Probability()
        prob.matrix = None
        
        with self.assertRaises(ValueError) as context:
            SuitsCanonicalTransformer(prob)
        self.assertIn("Invalid probability matrix", str(context.exception))
    
    def test_suit_strength_calculation(self):
        """Test that suit strengths are calculated correctly."""
        transformer = SuitsCanonicalTransformer(self.prob)
        strengths = transformer.get_suit_strengths()
        
        # Verify we have strengths for all suits
        self.assertEqual(len(strengths), 4)
        
        # Suit 2 should be strongest (highest probabilities)
        # Suit 0 should be weakest (lowest probabilities)
        self.assertGreater(strengths[2], strengths[1])
        self.assertGreater(strengths[1], strengths[3])
        self.assertGreater(strengths[3], strengths[0])
        
        # All strengths should be positive
        for strength in strengths.values():
            self.assertGreaterEqual(strength, 0)
    
    def test_canonical_mapping_order(self):
        """Test that canonical mapping orders suits by strength correctly."""
        transformer = SuitsCanonicalTransformer(self.prob)
        mapping = transformer.get_mapping()
        reverse_mapping = transformer.get_reverse_mapping()
        strengths = transformer.get_suit_strengths()
        
        # Verify mapping is complete
        self.assertEqual(len(mapping), 4)
        self.assertEqual(len(reverse_mapping), 4)
        
        # Check that strongest suit maps to canonical index 0
        strongest_suit = max(strengths.keys(), key=lambda x: strengths[x])
        self.assertEqual(mapping[strongest_suit], 0)
        
        # Check that weakest suit maps to canonical index 3
        weakest_suit = min(strengths.keys(), key=lambda x: strengths[x])
        self.assertEqual(mapping[weakest_suit], 3)
        
        # Verify reverse mapping consistency
        for original, canonical in mapping.items():
            self.assertEqual(reverse_mapping[canonical], original)
    
    def test_transform_function(self):
        """Test individual suit transformation."""
        transformer = SuitsCanonicalTransformer(self.prob)
        
        # Test valid transformations
        for suit in range(4):
            canonical = transformer.transform(suit)
            self.assertIn(canonical, range(4))
            
            # Test reverse transformation
            original = transformer.reverse(canonical)
            self.assertEqual(original, suit)
    
    def test_transform_invalid_indices(self):
        """Test transformation with invalid indices raises ValueError."""
        transformer = SuitsCanonicalTransformer(self.prob)
        
        # Test invalid suit indices for transform
        with self.assertRaises(ValueError):
            transformer.transform(-1)
        with self.assertRaises(ValueError):
            transformer.transform(4)
        
        # Test invalid canonical indices for reverse
        with self.assertRaises(ValueError):
            transformer.reverse(-1)
        with self.assertRaises(ValueError):
            transformer.reverse(4)
    
    def test_get_transform_function(self):
        """Test getting lambda function for transformation."""
        transformer = SuitsCanonicalTransformer(self.prob)
        transform_func = transformer.get_transform_function()
        
        # Test that lambda function works correctly
        for suit in range(4):
            self.assertEqual(transform_func(suit), transformer.transform(suit))
    
    def test_get_reverse_function(self):
        """Test getting lambda function for reverse transformation."""
        transformer = SuitsCanonicalTransformer(self.prob)
        reverse_func = transformer.get_reverse_function()
        
        # Test that lambda function works correctly
        for canonical in range(4):
            self.assertEqual(reverse_func(canonical), transformer.reverse(canonical))
    
    def test_apply_to_probability_default(self):
        """Test applying transformation to default probability (self)."""
        transformer = SuitsCanonicalTransformer(self.prob)
        original_matrix = np.copy(self.prob.matrix)
        
        # Apply transformation
        result = transformer.apply_to_probability()
        
        # Check that method returns the probability object
        self.assertIs(result, self.prob)
        
        # Check that matrix was modified
        self.assertFalse(np.array_equal(self.prob.matrix, original_matrix))
        
        # Verify transformation logic by checking a specific element
        mapping = transformer.get_mapping()
        for player in range(4):
            for rank in range(8):
                for original_suit, canonical_suit in mapping.items():
                    self.assertAlmostEqual(
                        self.prob.matrix[player, canonical_suit, rank],
                        original_matrix[player, original_suit, rank],
                        places=5
                    )
    
    def test_apply_to_probability_custom(self):
        """Test applying transformation to custom probability object."""
        transformer = SuitsCanonicalTransformer(self.prob)
        
        # Create a different probability object
        other_prob = Probability()
        other_prob.matrix = np.random.rand(4, 4, 8).astype(np.float32)
        original_matrix = np.copy(other_prob.matrix)
        
        # Apply transformation
        result = transformer.apply_to_probability(other_prob)
        
        # Check that method returns the probability object
        self.assertIs(result, other_prob)
        
        # Check that matrix was modified
        self.assertFalse(np.array_equal(other_prob.matrix, original_matrix))
    
    def test_reverse_probability_default(self):
        """Test reverse transformation on default probability (self)."""
        transformer = SuitsCanonicalTransformer(self.prob)
        original_matrix = np.copy(self.prob.matrix)
        
        # Apply and then reverse transformation
        transformer.apply_to_probability()
        transformer.reverse_probability()
        
        # Matrix should be back to original state
        np.testing.assert_array_almost_equal(self.prob.matrix, original_matrix, decimal=5)
    
    def test_reverse_probability_custom(self):
        """Test reverse transformation on custom probability object."""
        transformer = SuitsCanonicalTransformer(self.prob)
        
        # Create a different probability object
        other_prob = Probability()
        other_prob.matrix = np.random.rand(4, 4, 8).astype(np.float32)
        original_matrix = np.copy(other_prob.matrix)
        
        # Apply and then reverse transformation
        transformer.apply_to_probability(other_prob)
        result = transformer.reverse_probability(other_prob)
        
        # Check that method returns the probability object
        self.assertIs(result, other_prob)
        
        # Matrix should be back to original state
        np.testing.assert_array_almost_equal(other_prob.matrix, original_matrix, decimal=5)
    
    def test_transformation_roundtrip(self):
        """Test that applying transformation and then reverse gets back original."""
        transformer = SuitsCanonicalTransformer(self.prob)
        original_matrix = np.copy(self.prob.matrix)
        
        # Apply transformation
        transformer.apply_to_probability()
        transformed_matrix = np.copy(self.prob.matrix)
        
        # Apply reverse transformation
        transformer.reverse_probability()
        
        # Should be back to original
        np.testing.assert_array_almost_equal(self.prob.matrix, original_matrix, decimal=5)
        
        # Verify transformation actually changed something
        self.assertFalse(np.array_equal(transformed_matrix, original_matrix))
    
    def test_mapping_consistency(self):
        """Test that mappings are consistent and bijective."""
        transformer = SuitsCanonicalTransformer(self.prob)
        mapping = transformer.get_mapping()
        reverse_mapping = transformer.get_reverse_mapping()
        
        # Test that mappings are bijective
        self.assertEqual(len(mapping), len(reverse_mapping))
        
        # Test that forward and reverse mappings are consistent
        for original, canonical in mapping.items():
            self.assertEqual(reverse_mapping[canonical], original)
        
        # Test that all indices are covered
        self.assertEqual(set(mapping.keys()), set(range(4)))
        self.assertEqual(set(mapping.values()), set(range(4)))
    
    def test_equal_strength_suits(self):
        """Test behavior when multiple suits have equal strength."""
        # Create probability with equal strengths
        equal_prob = Probability()
        equal_prob.matrix.fill(0.25)  # All equal probabilities
        
        transformer = SuitsCanonicalTransformer(equal_prob)
        strengths = transformer.get_suit_strengths()
        
        # All strengths should be equal
        strength_values = list(strengths.values())
        for i in range(1, len(strength_values)):
            self.assertAlmostEqual(strength_values[0], strength_values[i], places=5)
        
        # Mapping should still be valid
        mapping = transformer.get_mapping()
        self.assertEqual(len(mapping), 4)
        self.assertEqual(set(mapping.values()), set(range(4)))
    
    def test_negative_probabilities(self):
        """Test behavior with negative probabilities (played cards)."""
        # Add some negative probabilities (played cards)
        self.prob.matrix[0, 1, 0] = -1.0  # Card played
        self.prob.matrix[1, 2, 3] = -1.0  # Another card played
        
        transformer = SuitsCanonicalTransformer(self.prob)
        strengths = transformer.get_suit_strengths()
        
        # Negative probabilities should not contribute to strength
        # (they're filtered out by np.maximum(0, ...))
        for strength in strengths.values():
            self.assertGreaterEqual(strength, 0)
        
        # Transformation should still work
        original_matrix = np.copy(self.prob.matrix)
        transformer.apply_to_probability()
        transformer.reverse_probability()
        np.testing.assert_array_almost_equal(self.prob.matrix, original_matrix, decimal=5)
    
    def test_str_representation(self):
        """Test string representation of the transformer."""
        transformer = SuitsCanonicalTransformer(self.prob)
        str_repr = str(transformer)
        
        # Should contain key information
        self.assertIn("SuitsCanonicalTransformer", str_repr)
        self.assertIn("Suit Strengths", str_repr)
        self.assertIn("Canonical Mapping", str_repr)
        
        # Should contain all suit indices
        for suit in range(4):
            self.assertIn(f"Suit {suit}", str_repr)
    
    def test_repr_representation(self):
        """Test repr representation of the transformer."""
        transformer = SuitsCanonicalTransformer(self.prob)
        repr_str = repr(transformer)
        
        # Should contain class name and mapping
        self.assertIn("SuitsCanonicalTransformer", repr_str)
        self.assertIn("mapping=", repr_str)
    
    @patch('src.states.probability.Probability.change_suits')
    def test_apply_calls_change_suits(self, mock_change_suits):
        """Test that apply_to_probability calls change_suits correctly."""
        mock_change_suits.return_value = self.prob
        
        transformer = SuitsCanonicalTransformer(self.prob)
        transformer.apply_to_probability()
        
        # Verify that change_suits was called
        mock_change_suits.assert_called_once()
        
        # Verify the function passed is callable
        args, kwargs = mock_change_suits.call_args
        self.assertTrue(callable(args[0]))


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(SuitsCanonicalTransformerTest)
    
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
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")