import torch
import time
import numpy as np
import os
from src.deck import Deck
from src.stages.player.actions import Action
from src.states.probability import Probability
from src.stages.player.actions import ActionCardPlay
from src.canonical.suits_canonical_transformer import SuitsCanonicalTransformer
from src.stages.player.actions import Action, ActionCardPlay
from src.stages.player.ppo.ppo_agent import PPOAgent
from src.stages.player.ppo.memory import PPOMemory

class PPOBeloteAgent:
    
    def __init__(self, network, memory: PPOMemory=None):
        self.memory = memory
        self.seed = int(time.time() * 1000) % (2**32 - 1)
        self.ppo = PPOAgent(network)
        self.env_index = 0

    def init(self, env, env_index=0, probability=None):
        # Store the environment index
        self.env_index = env_index
        
        # Initialize Probability
        self.init_probability(env, probability)
    
    def init_probability(self, env, probability=None):
        # Initialize the probability matrix for the agent    
        self.probability = probability if probability else Probability()

        # Reset My Hands Probability
        all_cards = Deck.new_cards()
        for card in all_cards:
            self.probability.update(0, card.suit, card.rank, 1 if card in env.deck.hands[self.env_index] else 0)

    def observe(self, player, action):
        # Every Agent observation is unique for themselves
        if isinstance(action, ActionCardPlay):
            self.probability.update(player, action.card.suit, action.card.rank, -1)
    
    def choose_action(self, env):
        # Get state representation
        transformer = SuitsCanonicalTransformer(self.probability) if self.probability else None
        transform = transformer.get_transform_function() if transformer else None
        
        # Get tensors (with error handling)
        probability_tensor = (self.probability.copy().change_suits(transform).to_tensor().unsqueeze(0))
        table_tensor = (env.table.copy().change_suits(transform).to_tensor().unsqueeze(0))
        trump_tensor = (env.trump.copy().change_suits(transform).to_tensor().unsqueeze(0))
        
        # Get valid actions
        valid_cards = env.valid_cards()
        valid_indices = []
        card_map = {}
        
        for card in valid_cards:
            transformed = card.copy().change_suits(transform)
            idx = transformed.rank * 4 + transformed.suit
                
            valid_indices.append(idx)
            card_map[idx] = card

        #action_idx, value, probability, table, trump, log_prob = self.ppo.act(
        experience = self.ppo.act(
            Action.TYPE_CARD_PLAY,
            probability_tensor, 
            table_tensor, 
            trump_tensor, 
            valid_indices,
            training=True if self.memory else False,
        )

        # Memorize the experience
        if self.memory is not None:
            self.memory.add_experience(experience)
        
        # Check if action_idx is valid
        action_idx = experience['action']
        if action_idx in valid_indices:
            if experience['action_type'] == Action.TYPE_CARD_PLAY:
                return ActionCardPlay(card_map[action_idx])
        
        raise ValueError(f"Invalid action: {action_idx}. Valid indices: {valid_indices}")
    
    def learn(self, batch_size=64):
        if self.memory is None:
            raise ValueError("Memory is not initialized. Cannot learn without memory.")
        
        # Cut the memory to keep only the last batch_size * 10 experiences
        self.memory.cut_experience(keep_n_from_end=batch_size * 10)

        # Get a random batch of experiences
        batch = self.memory.random_batch(batch_size)
        if batch is None:
            return
        
        # Perform learning
        self.ppo.learn(batch)

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