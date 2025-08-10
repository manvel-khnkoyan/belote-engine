import unittest
from unittest.mock import Mock, patch, call
import torch

# Import the classes under test
from src.stages.player.ppo.belote_agent import PPOBeloteAgent
from src.stages.player.actions import ActionCardPlay
from src.card import Card

class PPOBeloteAgentTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock network with parameters to satisfy PPOAgent
        self.mock_network = Mock()
        dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.mock_network.parameters.return_value = [dummy_param]
        self.mock_network.to.return_value = self.mock_network
        
        # Create agent instance
        self.agent = PPOBeloteAgent(self.mock_network)

    def test_observe_with_action_card_play(self):
        """Test observe method with ActionCardPlay updates probability correctly."""
        # Setup probability mock
        self.agent.probability = Mock()
        
        # Create ActionCardPlay
        mock_card = Mock()
        mock_card.suit = 2
        mock_card.rank = 5
        action = ActionCardPlay(mock_card)
        
        # Call observe
        self.agent.observe(player=1, action=action)
        
        # Verify probability update was called with correct parameters
        self.agent.probability.update.assert_called_once_with(1, 2, 5, -1)

    def test_observe_with_different_player_and_card(self):
        """Test observe method with different player and card values."""
        # Setup probability mock
        self.agent.probability = Mock()
        
        # Create ActionCardPlay with different card
        mock_card = Mock()
        mock_card.suit = 0
        mock_card.rank = 7
        action = ActionCardPlay(mock_card)
        
        # Call observe with different player
        self.agent.observe(player=3, action=action)
        
        # Verify probability update was called with correct parameters
        self.agent.probability.update.assert_called_once_with(3, 0, 7, -1)

    def test_observe_with_non_card_action(self):
        """Test observe method ignores non-ActionCardPlay actions."""
        # Setup probability mock
        self.agent.probability = Mock()
        
        # Create non-ActionCardPlay action
        mock_action = Mock(spec=['some_other_method'])
        
        # Call observe
        self.agent.observe(player=2, action=mock_action)
        
        # Verify probability update was NOT called
        self.agent.probability.update.assert_not_called()

    def test_observe_multiple_card_actions(self):
        """Test observe method handles multiple ActionCardPlay calls correctly."""
        # Setup probability mock
        self.agent.probability = Mock()
        
        # Create multiple ActionCardPlay actions
        mock_card1 = Mock()
        mock_card1.suit = 1
        mock_card1.rank = 3
        action1 = ActionCardPlay(mock_card1)
        
        mock_card2 = Mock()
        mock_card2.suit = 3
        mock_card2.rank = 6
        action2 = ActionCardPlay(mock_card2)
        
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
        """Test observe method behavior when probability is None."""
        # Set probability to None
        self.agent.probability = None
        
        # Create ActionCardPlay
        mock_card = Mock()
        mock_card.suit = 1
        mock_card.rank = 4
        action = ActionCardPlay(mock_card)
        
        # This should raise an AttributeError when trying to call update on None
        with self.assertRaises(AttributeError):
            self.agent.observe(player=1, action=action)

    @patch('src.stages.player.ppo.belote_agent.PPOMemory')
    def test_initialization_with_memory(self, mock_memory_class):
        """Test agent initialization with memory parameter."""
        mock_memory = Mock()
        agent = PPOBeloteAgent(self.mock_network, memory=mock_memory)
        
        self.assertEqual(agent.memory, mock_memory)
        self.assertIsNotNone(agent.seed)
        self.assertIsNotNone(agent.ppo)

    def test_initialization_without_memory(self):
        """Test agent initialization without memory parameter."""
        agent = PPOBeloteAgent(self.mock_network)
        
        self.assertIsNone(agent.memory)
        self.assertIsNotNone(agent.seed)
        self.assertIsNotNone(agent.ppo)

    @patch('src.stages.player.ppo.belote_agent.Probability')
    def test_init_probability_with_provided_probability(self, mock_probability_class):
        """Test init_probability method when probability is provided."""
        # Create mock environment
        mock_env = Mock()
        mock_env.deck.hands = {0: []}  # Empty hand for simplicity
        
        # Create mock probability
        mock_probability = Mock()
        
        # Initialize with provided probability
        self.agent.init_probability(mock_env, probability=mock_probability)
        
        # Verify the provided probability is used
        self.assertEqual(self.agent.probability, mock_probability)

    @patch('src.stages.player.ppo.belote_agent.Probability')
    def test_init_probability_without_provided_probability(self, mock_probability_class):
        """Test init_probability method when no probability is provided."""
        mock_probability_instance = Mock()
        mock_probability_class.return_value = mock_probability_instance
        
        # Create mock environment with cards
        card = Card(1, 5)
        mock_env = Mock()
        mock_env.deck.hands = {0: [card]}
        self.agent.env_index = 0
        
        # Initialize without provided probability
        self.agent.init_probability(mock_env)
        
        # Verify new Probability instance is created and used
        mock_probability_class.assert_called_once()
        self.assertEqual(self.agent.probability, mock_probability_instance)
        
        # Check that update was called multiple times (for all cards in deck)
        assert mock_probability_instance.update.call_count == 32
        
        # Verify that the specific card in hand was updated correctly
        calls_with_value_1 = [
            call for call in mock_probability_instance.update.call_args_list
            if call[0][0] == 0 and call[0][1] == card.suit and call[0][2] == card.rank and call[0][3] == 1
        ]
        
        # Should have exactly one call with value=1 (the card in hand)
        assert len(calls_with_value_1) == 1
        # And it should be for our specific card
        assert calls_with_value_1[0] == call(0, 1, 5, 1)

    def test_learn_without_memory_raises_error(self):
        """Test that learn method raises ValueError when memory is None."""
        self.agent.memory = None
        
        with self.assertRaises(ValueError) as context:
            self.agent.learn()
        
        self.assertIn("Memory is not initialized", str(context.exception))

    def test_learn_with_memory_no_batch(self):
        """Test learn method when memory returns no batch."""
        mock_memory = Mock()
        mock_memory.random_batch.return_value = None
        self.agent.memory = mock_memory
        
        # Should not raise error and should call memory methods
        self.agent.learn(batch_size=32)
        
        mock_memory.cut_experience.assert_called_once_with(keep_n_from_end=320)
        mock_memory.random_batch.assert_called_once_with(32)

    def test_learn_with_memory_and_batch(self):
        """Test learn method when memory returns a valid batch."""
        mock_memory = Mock()
        mock_batch = {'some': 'data'}
        mock_memory.random_batch.return_value = mock_batch
        self.agent.memory = mock_memory
        
        # Mock the PPO agent's learn method
        self.agent.ppo = Mock()
        
        # Call learn
        self.agent.learn(batch_size=16)
        
        # Verify memory methods were called
        mock_memory.cut_experience.assert_called_once_with(keep_n_from_end=160)
        mock_memory.random_batch.assert_called_once_with(16)
        
        # Verify PPO learn was called with the batch
        self.agent.ppo.learn.assert_called_once_with(mock_batch)


if __name__ == '__main__':
    unittest.main(verbosity=2)