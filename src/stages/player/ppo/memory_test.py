import unittest
from unittest.mock import patch
import numpy as np
import time

# Import the class under test
from src.stages.player.ppo.memory import PPOMemory


class PPOMemoryTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.memory = PPOMemory()
        
        # Sample experience for testing
        self.sample_experience = {
            'action_type': 1,
            'action': 5,
            'probability': 0.8,
            'table': [1, 2, 3],
            'trump': [0, 1],
            'value': 0.5,
            'log_prob': -0.2
        }

    def test_initialization(self):
        """Test PPOMemory initialization."""
        memory = PPOMemory()
        
        # Check that all lists are initialized as empty
        self.assertEqual(len(memory.action_types), 0)
        self.assertEqual(len(memory.actions), 0)
        self.assertEqual(len(memory.probabilities), 0)
        self.assertEqual(len(memory.tables), 0)
        self.assertEqual(len(memory.trumps), 0)
        self.assertEqual(len(memory.values), 0)
        self.assertEqual(len(memory.log_probs), 0)
        self.assertEqual(len(memory.rewards), 0)
        
        # Check that seed is set
        self.assertIsInstance(memory.seed, int)
        self.assertGreater(memory.seed, 0)

    def test_clear(self):
        """Test clear method resets all memory lists."""
        # Add some experiences first
        self.memory.add_experience(self.sample_experience)
        self.memory.add_experience(self.sample_experience)
        
        # Verify experiences were added
        self.assertEqual(len(self.memory.actions), 2)
        
        # Clear memory
        self.memory.clear()
        
        # Verify all lists are empty
        self.assertEqual(len(self.memory.action_types), 0)
        self.assertEqual(len(self.memory.actions), 0)
        self.assertEqual(len(self.memory.probabilities), 0)
        self.assertEqual(len(self.memory.tables), 0)
        self.assertEqual(len(self.memory.trumps), 0)
        self.assertEqual(len(self.memory.values), 0)
        self.assertEqual(len(self.memory.log_probs), 0)
        self.assertEqual(len(self.memory.rewards), 0)

    def test_add_experience(self):
        """Test adding a single experience to memory."""
        self.memory.add_experience(self.sample_experience)
        
        # Verify all fields were added correctly
        self.assertEqual(len(self.memory.actions), 1)
        self.assertEqual(self.memory.action_types[0], 1)
        self.assertEqual(self.memory.actions[0], 5) 
        self.assertEqual(self.memory.probabilities[0], 0.8)
        self.assertEqual(self.memory.tables[0], [1, 2, 3])
        self.assertEqual(self.memory.trumps[0], [0, 1])
        self.assertEqual(self.memory.values[0], 0.5)
        self.assertEqual(self.memory.log_probs[0], -0.2)
        self.assertEqual(self.memory.rewards[0], 0.0)  # Initialized to 0.0

    def test_add_multiple_experiences(self):
        """Test adding multiple experiences to memory."""
        experience2 = {
            'action_type': 2,
            'action': 10,
            'probability': 0.6,
            'table': [4, 5, 6],
            'trump': [1, 0],
            'value': 0.3,
            'log_prob': -0.5
        }
        
        self.memory.add_experience(self.sample_experience)
        self.memory.add_experience(experience2)
        
        # Verify both experiences were added
        self.assertEqual(len(self.memory.actions), 2)
        self.assertEqual(self.memory.actions[0], 5)
        self.assertEqual(self.memory.actions[1], 10)
        self.assertEqual(self.memory.probabilities[0], 0.8)
        self.assertEqual(self.memory.probabilities[1], 0.6)
        
        # Verify rewards are initialized to 0.0
        self.assertEqual(self.memory.rewards[0], 0.0)
        self.assertEqual(self.memory.rewards[1], 0.0)

    def test_cut_experience_keep_from_end(self):
        """Test cutting experience to keep only last N entries."""
        # Add 5 experiences
        for i in range(5):
            exp = self.sample_experience.copy()
            exp['action'] = i
            self.memory.add_experience(exp)
        
        # Keep only last 3
        self.memory.cut_experience(keep_n_from_end=3)
        
        # Verify only last 3 remain
        self.assertEqual(len(self.memory.actions), 3)
        self.assertEqual(self.memory.actions, [2, 3, 4])

    def test_cut_experience_keep_more_than_available(self):
        """Test cutting experience when keeping more than available."""
        # Add 3 experiences
        for i in range(3):
            exp = self.sample_experience.copy()
            exp['action'] = i
            self.memory.add_experience(exp)
        
        # Try to keep 5 (more than available)
        self.memory.cut_experience(keep_n_from_end=5)
        
        # All 3 should remain
        self.assertEqual(len(self.memory.actions), 3)
        self.assertEqual(self.memory.actions, [0, 1, 2])

    def test_cut_experience_zero_or_negative(self):
        """Test cutting experience with zero or negative values."""
        # Add experiences
        self.memory.add_experience(self.sample_experience)
        self.memory.add_experience(self.sample_experience)
        
        original_length = len(self.memory.actions)
        
        # Test with zero
        self.memory.cut_experience(keep_n_from_end=0)
        self.assertEqual(len(self.memory.actions), original_length)
        
        # Test with negative
        self.memory.cut_experience(keep_n_from_end=-1)
        self.assertEqual(len(self.memory.actions), original_length)

    def test_cut_experience_empty_memory(self):
        """Test cutting experience on empty memory."""
        # Should not raise error
        self.memory.cut_experience(keep_n_from_end=5)
        self.assertEqual(len(self.memory.actions), 0)

    def test_random_batch_empty_memory(self):
        """Test random_batch returns None for empty memory."""
        result = self.memory.random_batch(batch_size=5)
        self.assertIsNone(result)

    def test_random_batch_normal_case(self):
        """Test random_batch returns correct batch structure."""
        # Add multiple experiences
        for i in range(10):
            exp = self.sample_experience.copy()
            exp['action'] = i
            exp['probability'] = i * 0.1
            self.memory.add_experience(exp)
        
        # Get a batch
        batch = self.memory.random_batch(batch_size=5)
        
        # Verify batch structure
        self.assertIsNotNone(batch)
        self.assertIn('action_types', batch)
        self.assertIn('actions', batch)
        self.assertIn('probabilities', batch)
        self.assertIn('tables', batch)
        self.assertIn('trumps', batch)
        self.assertIn('values', batch)
        self.assertIn('log_probs', batch)
        self.assertIn('rewards', batch)
        
        # Verify batch size
        self.assertEqual(len(batch['actions']), 5)
        self.assertEqual(len(batch['probabilities']), 5)

    def test_random_batch_larger_than_memory(self):
        """Test random_batch when requested size > memory size."""
        # Add 3 experiences
        for i in range(3):
            exp = self.sample_experience.copy()
            exp['action'] = i
            self.memory.add_experience(exp)
        
        # Request batch of 5
        batch = self.memory.random_batch(batch_size=5)
        
        # Should return all 3 available
        self.assertEqual(len(batch['actions']), 3)
        self.assertEqual(set(batch['actions']), {0, 1, 2})

    @patch('numpy.random.default_rng')
    def test_random_batch_randomness(self, mock_rng):
        """Test that random_batch uses proper random number generation."""
        # Setup mock
        mock_rng_instance = mock_rng.return_value
        mock_rng_instance.choice.return_value = np.array([0, 2, 4])
        
        # Add experiences
        for i in range(5):
            exp = self.sample_experience.copy()
            exp['action'] = i
            self.memory.add_experience(exp)
        
        # Get batch
        batch = self.memory.random_batch(batch_size=3)
        
        # Verify RNG was called correctly
        mock_rng.assert_called_once()
        mock_rng_instance.choice.assert_called_once_with(5, size=3, replace=False)
        
        # Verify correct indices were used
        self.assertEqual(batch['actions'], [0, 2, 4])

    def test_updated_last_rewards_single_step(self):
        """Test updating rewards for the last experience."""
        # Add experiences
        for i in range(3):
            self.memory.add_experience(self.sample_experience)
        
        # Update last reward
        self.memory.updated_last_rewards(reward_value=10.0, last_n=1)
        
        # Only last reward should be updated
        expected_rewards = [0.0, 0.0, 10.0]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_updated_last_rewards_multiple_steps(self):
        """Test updating rewards for multiple last experiences with decay."""
        # Add experiences
        for i in range(4):
            self.memory.add_experience(self.sample_experience)
        
        # Update last 3 rewards with decay
        self.memory.updated_last_rewards(reward_value=8.0, last_n=3, decay_factor=2)
        
        # Expected: last gets 8.0/1, second-to-last gets 8.0/2, third-to-last gets 8.0/4
        expected_rewards = [0.0, 2.0, 4.0, 8.0]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_updated_last_rewards_exceeds_memory_size(self):
        """Test updating rewards when last_n exceeds memory size."""
        # Add 2 experiences
        for i in range(2):
            self.memory.add_experience(self.sample_experience)
        
        # Try to update last 5 (more than available)
        self.memory.updated_last_rewards(reward_value=6.0, last_n=5, decay_factor=2)
        
        # Should update both available rewards
        # Index 1 gets 6.0/1, Index 0 gets 6.0/2
        expected_rewards = [3.0, 6.0]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_updated_last_rewards_empty_memory(self):
        """Test updating rewards on empty memory."""
        # Should not raise error
        self.memory.updated_last_rewards(reward_value=5.0, last_n=2)
        self.assertEqual(len(self.memory.rewards), 0)

    def test_updated_last_rewards_zero_reward(self):
        """Test updating rewards with zero reward value."""
        self.memory.add_experience(self.sample_experience)
        
        # Update with zero reward
        self.memory.updated_last_rewards(reward_value=0.0, last_n=1)
        
        # Reward should remain 0.0
        self.assertEqual(self.memory.rewards[0], 0.0)

    def test_updated_last_rewards_negative_reward(self):
        """Test updating rewards with negative reward value."""
        self.memory.add_experience(self.sample_experience)
        
        # Update with negative reward
        self.memory.updated_last_rewards(reward_value=-5.0, last_n=1)
        
        # Reward should be negative
        self.assertEqual(self.memory.rewards[0], -5.0)

    def test_updated_last_rewards_negative_multiple_steps(self):
        """Test updating multiple rewards with negative values and decay."""
        # Add 4 experiences
        for i in range(4):
            self.memory.add_experience(self.sample_experience)
        
        # Update last 3 with negative reward
        self.memory.updated_last_rewards(reward_value=-12.0, last_n=3, decay_factor=2)
        
        # Expected: last gets -12.0/1, second-to-last gets -12.0/2, third-to-last gets -12.0/4
        expected_rewards = [0.0, -3.0, -6.0, -12.0]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_updated_last_rewards_mixed_positive_negative(self):
        """Test multiple reward updates with both positive and negative values."""
        # Add 3 experiences
        for i in range(3):
            self.memory.add_experience(self.sample_experience)
        
        # First update with positive reward
        self.memory.updated_last_rewards(reward_value=6.0, last_n=2, decay_factor=2)
        # Expected after first update: [0.0, 3.0, 6.0]
        
        # Second update with negative reward
        self.memory.updated_last_rewards(reward_value=-4.0, last_n=1)
        # Expected after second update: [0.0, 3.0, 2.0] (6.0 + (-4.0))
        
        expected_rewards = [0.0, 3.0, 2.0]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_updated_last_rewards_negative_with_existing_positive(self):
        """Test negative reward updates on experiences that already have positive rewards."""
        # Add experiences and set some initial positive rewards
        for i in range(3):
            self.memory.add_experience(self.sample_experience)
        
        # Manually set some positive rewards
        self.memory.rewards = [1.0, 2.0, 3.0]
        
        # Apply negative reward update
        self.memory.updated_last_rewards(reward_value=-8.0, last_n=2, decay_factor=2)
        
        # Expected: rewards[1] = 2.0 + (-8.0/2) = -2.0, rewards[2] = 3.0 + (-8.0/1) = -5.0
        expected_rewards = [1.0, -2.0, -5.0]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_updated_last_rewards_large_negative_values(self):
        """Test updating rewards with large negative values."""
        # Add experiences
        for i in range(2):
            self.memory.add_experience(self.sample_experience)
        
        # Update with large negative reward
        self.memory.updated_last_rewards(reward_value=-100.0, last_n=2, decay_factor=3)
        
        # Expected: rewards[0] = 0.0 + (-100.0/3) ≈ -33.33, rewards[1] = 0.0 + (-100.0/1) = -100.0
        expected_rewards = [0.0 + (-100.0/3), 0.0 + (-100.0/1)]
        self.assertAlmostEqual(self.memory.rewards[0], expected_rewards[0], places=5)
        self.assertEqual(self.memory.rewards[1], expected_rewards[1])

    def test_updated_last_rewards_fractional_negative(self):
        """Test updating rewards with fractional negative values."""
        # Add experiences
        for i in range(3):
            self.memory.add_experience(self.sample_experience)
        
        # Update with fractional negative reward
        self.memory.updated_last_rewards(reward_value=-2.5, last_n=3, decay_factor=2)
        
        # Expected: 
        # rewards[0] = 0.0 + (-2.5/4) = -0.625
        # rewards[1] = 0.0 + (-2.5/2) = -1.25  
        # rewards[2] = 0.0 + (-2.5/1) = -2.5
        expected_rewards = [-0.625, -1.25, -2.5]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_updated_last_rewards_negative_different_decay_factors(self):
        """Test negative rewards with different decay factors."""
        # Add experiences
        for i in range(4):
            self.memory.add_experience(self.sample_experience)
        
        # Test with decay factor of 1 (no decay)
        self.memory.updated_last_rewards(reward_value=-6.0, last_n=2, decay_factor=1)
        
        # Expected: both last rewards get full negative value
        expected_rewards = [0.0, 0.0, -6.0, -6.0]
        self.assertEqual(self.memory.rewards, expected_rewards)
        
        # Reset rewards
        self.memory.rewards = [0.0, 0.0, 0.0, 0.0]
        
        # Test with large decay factor
        self.memory.updated_last_rewards(reward_value=-10.0, last_n=3, decay_factor=5)
        
        # Expected:
        # rewards[1] = 0.0 + (-10.0/25) = -0.4
        # rewards[2] = 0.0 + (-10.0/5) = -2.0
        # rewards[3] = 0.0 + (-10.0/1) = -10.0
        expected_rewards = [0.0, -0.4, -2.0, -10.0]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_updated_last_rewards_accumulating_negative_rewards(self):
        """Test multiple negative reward updates that accumulate."""
        # Add experiences
        for i in range(2):
            self.memory.add_experience(self.sample_experience)
        
        # First negative update
        self.memory.updated_last_rewards(reward_value=-3.0, last_n=1)
        self.assertEqual(self.memory.rewards, [0.0, -3.0])
        
        # Second negative update
        self.memory.updated_last_rewards(reward_value=-2.0, last_n=1)
        self.assertEqual(self.memory.rewards, [0.0, -5.0])  # -3.0 + (-2.0)
        
        # Third negative update affecting both
        self.memory.updated_last_rewards(reward_value=-4.0, last_n=2, decay_factor=2)
        # Expected: rewards[0] = 0.0 + (-4.0/2) = -2.0, rewards[1] = -5.0 + (-4.0/1) = -9.0
        expected_rewards = [-2.0, -9.0]
        self.assertEqual(self.memory.rewards, expected_rewards)

    def test_len_method(self):
        """Test __len__ method returns correct count."""
        # Initially empty
        self.assertEqual(len(self.memory), 0)
        
        # Add experiences
        self.memory.add_experience(self.sample_experience)
        self.assertEqual(len(self.memory), 1)
        
        self.memory.add_experience(self.sample_experience)
        self.assertEqual(len(self.memory), 2)

    def test_is_empty_method(self):
        """Test is_empty method."""
        # Initially empty
        self.assertTrue(self.memory.is_empty())
        
        # Add experience
        self.memory.add_experience(self.sample_experience)
        self.assertFalse(self.memory.is_empty())
        
        # Clear and check again
        self.memory.clear()
        self.assertTrue(self.memory.is_empty())

    def test_seed_increments_on_batch_calls(self):
        """Test that seed increments on random_batch calls."""
        # Add experiences
        for i in range(5):
            self.memory.add_experience(self.sample_experience)
        
        original_seed = self.memory.seed
        
        # Call random_batch
        self.memory.random_batch(batch_size=2)
        self.assertEqual(self.memory.seed, original_seed + 1)
        
        # Call again
        self.memory.random_batch(batch_size=2)
        self.assertEqual(self.memory.seed, original_seed + 2)

    def test_memory_consistency_after_operations(self):
        """Test that all memory lists maintain consistent length after operations."""
        # Add experiences
        for i in range(5):
            exp = self.sample_experience.copy()
            exp['action'] = i
            self.memory.add_experience(exp)
        
        # Verify all lists have same length
        lengths = [
            len(self.memory.action_types),
            len(self.memory.actions),
            len(self.memory.probabilities),
            len(self.memory.tables),
            len(self.memory.trumps),
            len(self.memory.values),
            len(self.memory.log_probs),
            len(self.memory.rewards)
        ]
        self.assertTrue(all(length == 5 for length in lengths))
        
        # Cut experience and verify consistency
        self.memory.cut_experience(keep_n_from_end=3)
        lengths = [
            len(self.memory.action_types),
            len(self.memory.actions),
            len(self.memory.probabilities),
            len(self.memory.tables),
            len(self.memory.trumps),
            len(self.memory.values),
            len(self.memory.log_probs),
            len(self.memory.rewards)
        ]
        self.assertTrue(all(length == 3 for length in lengths))


if __name__ == '__main__':
    unittest.main(verbosity=2)