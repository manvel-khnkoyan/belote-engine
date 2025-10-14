import torch
import time
import numpy as np
import os
from src.deck import Deck
from src.stages.player.actions import Action # Assuming Action class defines TYPE_CARD_PLAY
from src.states.probability import Probability
from src.stages.player.actions import ActionPlayCard # Corrected to ActionPlayCard
from src.canonical.suits_canonical_transformer import SuitsCanonicalTransformer
from src.stages.player.ppo.ppo_agent import PPOAgent
from stages.player.agents.position0.network import BeloteNetwork

class PlayerAgent:
    
    def __init__(self, network, memory=None): # memory type hint removed for flexibility
        self.memory = memory
        self.seed = int(time.time() * 1000) % (2**32 - 1)
        self.ppo = PPOAgent(network) # PPOAgent instantiated here
        self.env_index = 0
        self.probability = None # Initialize probability to None

    def init(self, env, env_index=0, probability=None):
        # Store the environment index
        self.env_index = env_index
        
        # Initialize Probability
        self._init_probability(env, probability) # Renamed to private helper

    def _init_probability(self, env, probability_state=None): # Renamed for clarity
        # Initialize the probability matrix for the agent    
        self.probability = probability_state if probability_state else Probability()

        # Reset My Hands Probability
        all_cards = Deck.new_cards()
        for card in all_cards:
            # Player 0 in probability matrix always represents 'self' from the agent's perspective
            has_card = 1 if card in env.deck.hands[self.env_index] else 0
            self.probability.update(0, card.suit, card.rank, has_card)

    def observe(self, player_idx, action):
        """
        Observes an action played by any player and updates the agent's probability matrix.
        """
        # Every Agent observation is unique for themselves
        # Rotate the probability matrix so that the observing player is 'player 0'
        # and the actual player who performed the action is correctly mapped.
        # This rotation ensures the agent's internal state is always from its own perspective.
        player_offset = (player_idx - self.env_index + 4) % 4
        
        if isinstance(action, ActionPlayCard):
            # When a card is played, it's no longer in anyone's hand (except if it was just played)
            # We set its probability to -1 across all players to mark it as played.
            # No need for player_offset here as -1 means 'played', not 'in hand'.
            for p in range(4):
                self.probability.update(p, action.card.suit, action.card.rank, -1)
            # If the action was by an opponent, we can also set their probability for that card to 0
            # (though -1 already implies this).
            # If the action was by 'self', its probability for self was already 1 before being played.

    def choose_action(self, env):
        # Create a copy of the probability matrix and canonicalize it from the agent's perspective
        agent_probability = self.probability.copy()
        transformer = SuitsCanonicalTransformer(agent_probability)
        transform_func = transformer.get_transform_function()
        
        # Apply canonical transformation to agent's probability, table, and trump state
        canonical_probability = agent_probability.change_suits(transform_func)
        canonical_table = env.table.copy().change_suits(transform_func)
        
        # Handle Trump object transformation
        # Your Trump class doesn't have change_suits method, so we need to handle it differently
        canonical_trump = self._canonicalize_trump(env.trump, transform_func)
        
        # Get valid actions for the *current* environment (before canonicalization)
        # and map them to their canonical indices.
        valid_cards_env = env.valid_cards()
        valid_indices = []
        card_map_to_original = {} # Map canonical index back to original card object

        for original_card in valid_cards_env:
            # Create a canonical version of the card to get its index
            canonical_card = original_card.copy().change_suits(transform_func)
            # Assuming card_to_id in network.py handles this:
            # idx = (is_trump * 32) + (suit * 8) + rank
            # For policy head, it's just suit*rank (0-31)
            # Assuming BeloteNetwork.total_card_types is 32 (4 suits * 8 ranks)
            # This index needs to match the output dimension of self.ppo.network.card_policy
            
            # The card_policy output uses a flat index from 0-31 (suit * 8 + rank)
            # We need to ensure this mapping is consistent with BeloteNetwork's output.
            canonical_card_idx = canonical_card.suit * self.ppo.network.num_ranks + canonical_card.rank
            
            valid_indices.append(canonical_card_idx)
            card_map_to_original[canonical_card_idx] = original_card # Map canonical index to original card

        experience = self.ppo.act(
            action_type=Action.TYPE_CARD_PLAY, # Pass the action type
            probability_state=canonical_probability, # Pass Probability object
            table_state=canonical_table,           # Pass Table object
            trump_state=canonical_trump,           # Pass Trump object
            valid_actions=valid_indices,
        )

        # Memorize the experience
        if self.memory is not None:
            self.memory.add_experience(experience)
        
        # Check if action_idx is valid
        action_idx = experience['action'] # This is the canonical action index
        if action_idx in valid_indices:
            if experience['action_type'] == Action.TYPE_CARD_PLAY:
                # Return the ORIGINAL card object corresponding to the chosen canonical index
                return ActionPlayCard(card_map_to_original[action_idx])
        
        raise ValueError(f"Agent chose an invalid action: {action_idx}. Valid canonical indices: {valid_indices}")
    
    def _canonicalize_trump(self, trump, transform_func):
        """
        Canonicalize a Trump object according to the transformation function.
        Since Trump class doesn't have change_suits method, we handle it manually.
        """
        from trump import Trump
        
        if trump.mode == Trump.NO_TRUMP:
            # For NO_TRUMP, return a copy as-is since suit doesn't matter
            canonical_trump = Trump(Trump.NO_TRUMP, None)
        else:
            # For REGULAR trump, transform the suit
            if trump.suit is not None:
                # Apply transformation to the trump suit
                # transform_func maps old suit -> new suit
                canonical_suit = transform_func(trump.suit)
                canonical_trump = Trump(Trump.REGULAR, canonical_suit)
            else:
                # Fallback if suit is None but mode is REGULAR
                canonical_trump = Trump(Trump.REGULAR, None)
        
        return canonical_trump
    
    def learn(self, samples):
        if not samples:
            raise ValueError("No valid samples provided for learning.")
        
        # Perform learning
        self.ppo.learn(samples)

    def save(self, path):
        torch.save({
            'network_state_dict': self.ppo.network.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
        }, path)

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.ppo.device)
            self.ppo.network.load_state_dict(checkpoint['network_state_dict'])
            self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return True
        return False