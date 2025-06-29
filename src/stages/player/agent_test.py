import unittest
from unittest.mock import Mock

# Import the classes under test
from src.stages.player.agent import PPOBeloteAgent
from src.stages.player.actions import ActionCardMove


class PPOBeloteAgentTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock network with at least one parameter to satisfy optimizer
        import torch
        self.mock_network = Mock()
        dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.mock_network.parameters.return_value = [dummy_param]
        self.mock_network.to.return_value = self.mock_network
        
        # Create agent instance
        self.agent = PPOBeloteAgent(self.mock_network, training=True)

    def test_updated_rewards_single_step(self):
        """Test updated_rewards with single step back."""
        # Setup rewards list
        self.agent.rewards = [1.0, 2.0, 3.0, 4.0]
        
        # Call method
        self.agent.updated_rewards(10.0, last_n=1)
        
        # Check that only the last reward was updated
        expected_rewards = [1.0, 2.0, 3.0, 14.0]  # 4.0 + 10.0/1
        self.assertEqual(self.agent.rewards, expected_rewards)

    def test_updated_rewards_multiple_steps(self):
        """Test updated_rewards with multiple steps back."""
        # Setup rewards list
        self.agent.rewards = [1.0, 2.0, 3.0, 4.0]
        
        # Call method
        self.agent.updated_rewards(8.0, last_n=3)
        
        # Check rewards with decay factor
        # Last: 4.0 + 8.0/1 = 12.0
        # Second to last: 3.0 + 8.0/2 = 7.0  
        # Third to last: 2.0 + 8.0/4 = 4.0
        expected_rewards = [1.0, 4.0, 7.0, 12.0]
        self.assertEqual(self.agent.rewards, expected_rewards)

    def test_updated_float_rewards_multiple_steps(self):
        """Test updated_rewards with float rewards and multiple steps back."""
        # Setup rewards list
        self.agent.rewards = [1.5, 2.5, 3.5, 4.5]
        
        # Call method
        self.agent.updated_rewards(5.3, last_n=3)
        
        # Check rewards with decay factor
        expected_rewards = [1.5, 3.825, 6.15, 9.8]
        self.assertEqual(self.agent.rewards, expected_rewards)

    def test_updated_rewards_last_n_exceeds_list_length(self):
        """Test updated_rewards when last_n exceeds rewards list length."""
        # Setup short rewards list
        self.agent.rewards = [1.0, 2.0]
        
        # Call method with last_n > list length
        self.agent.updated_rewards(6.0, last_n=5)
        
        # Should only update existing rewards
        expected_rewards = [4.0, 8.0]  # 1.0 + 6.0/4, 2.0 + 6.0/1
        self.assertEqual(self.agent.rewards, expected_rewards)

    def test_updated_rewards_empty_list(self):
        """Test updated_rewards with empty rewards list."""
        self.agent.rewards = []
        
        # Should not raise error
        self.agent.updated_rewards(5.0, last_n=1)
        
        # List should remain empty
        self.assertEqual(self.agent.rewards, [])

    def test_updated_rewards_zero_reward(self):
        """Test updated_rewards with zero reward value."""
        # Setup rewards list
        self.agent.rewards = [1.0, 2.0, 3.0]
        
        # Call method with zero reward
        self.agent.updated_rewards(0.0, last_n=2)
        
        # Rewards should remain unchanged
        expected_rewards = [1.0, 2.0, 3.0]
        self.assertEqual(self.agent.rewards, expected_rewards)

    def test_updated_rewards_negative_reward(self):
        """Test updated_rewards with negative reward value."""
        # Setup rewards list
        self.agent.rewards = [1.0, 2.0, 3.0]
        
        # Call method with negative reward
        self.agent.updated_rewards(-4.0, last_n=2)
        
        # Check rewards with negative values
        # Last: 3.0 + (-4.0)/1 = -1.0
        # Second to last: 2.0 + (-4.0)/2 = 0.0
        expected_rewards = [1.0, 0.0, -1.0]
        self.assertEqual(self.agent.rewards, expected_rewards)

    def test_observe_with_action_card_move(self):
        """Test observe method with ActionCardMove."""
        # Setup probability mock
        self.agent.probability = Mock()
        
        # Create ActionCardMove
        mock_card = Mock()
        mock_card.suit = 2
        mock_card.rank = 5
        action = ActionCardMove(mock_card)
        
        # Call observe
        self.agent.observe(player=1, action=action)
        
        # Verify probability update was called with correct parameters
        self.agent.probability.update.assert_called_once_with(1, 2, 5, -1)

    def test_observe_with_different_player_and_card(self):
        """Test observe method with different player and card."""
        # Setup probability mock
        self.agent.probability = Mock()
        
        # Create ActionCardMove with different card
        mock_card = Mock()
        mock_card.suit = 0
        mock_card.rank = 7
        action = ActionCardMove(mock_card)
        
        # Call observe with different player
        self.agent.observe(player=3, action=action)
        
        # Verify probability update was called with correct parameters
        self.agent.probability.update.assert_called_once_with(3, 0, 7, -1)

    def test_observe_with_non_card_action(self):
        """Test observe method with non-ActionCardMove action."""
        # Setup probability mock
        self.agent.probability = Mock()
        
        # Create non-ActionCardMove action
        mock_action = Mock(spec=['some_other_method'])
        
        # Call observe
        self.agent.observe(player=2, action=mock_action)
        
        # Verify probability update was NOT called
        self.agent.probability.update.assert_not_called()

    def test_observe_multiple_actions(self):
        """Test observe method called multiple times."""
        # Setup probability mock
        self.agent.probability = Mock()
        
        # Create multiple ActionCardMove actions
        mock_card1 = Mock()
        mock_card1.suit = 1
        mock_card1.rank = 3
        action1 = ActionCardMove(mock_card1)
        
        mock_card2 = Mock()
        mock_card2.suit = 3
        mock_card2.rank = 6
        action2 = ActionCardMove(mock_card2)
        
        # Call observe multiple times
        self.agent.observe(player=0, action=action1)
        self.agent.observe(player=2, action=action2)
        
        # Verify probability update was called twice with correct parameters
        expected_calls = [
            unittest.mock.call(0, 1, 3, -1),
            unittest.mock.call(2, 3, 6, -1)
        ]
        self.agent.probability.update.assert_has_calls(expected_calls)
        self.assertEqual(self.agent.probability.update.call_count, 2)

    def test_observe_without_probability_initialized(self):
        """Test observe when probability is not initialized."""
        # Don't initialize probability (it should be None by default)
        self.agent.probability = None
        
        # Create ActionCardMove
        mock_card = Mock()
        mock_card.suit = 1
        mock_card.rank = 4
        action = ActionCardMove(mock_card)
        
        # This should raise an AttributeError when trying to call update on None
        with self.assertRaises(AttributeError):
            self.agent.observe(player=1, action=action)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)