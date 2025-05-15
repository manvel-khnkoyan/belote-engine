import numpy as np
from src.stages.player.actions import ActionCardMove

# Define
class Randomer:
    def __init__(self):
        self.name = 'Random'
        self.env_index = 0
        self.seed=1

    def init(self, env, env_index=0):
        self.env_index = env_index

    def observe(self, player, action):
       # Do nothing
       return

    def choose_action(self, env):
        # Choose a random valid card
        valid_cards = env.valid_cards()
        
        if not valid_cards:
            raise ValueError("No valid cards for player")
        
        # Use numpy's random choice
        self.seed += 1
        rng = np.random.default_rng(seed=self.seed)
        card = rng.choice(valid_cards)
        
        return ActionCardMove(card)
