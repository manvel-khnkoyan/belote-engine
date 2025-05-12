import os
import time
import numpy as np
import json
from src.deck import Deck
from src.card import Card
from src.states.trump import Trump
from src.stages.player.env import BeloteEnv
from src.types import ACTION_TYPE_CARD

class History:
    def __init__(self, env):
        self.seed = int(time.time()) % 1000
        self.step = 0
        self.actions = []
        # Initialize the environment
        self.env = None
        # Snapshot states of the game
        self.trump = env.trump.copy()
        self.deck = env.deck.copy()
        self.next_player = env.next_player

    def reset(self):
        self.step = 0

        return BeloteEnv(self.deck.copy(), self.trump.copy(), self.next_player)

    def record(self, player, action):
        if self.step == 0 and self.actions:
            self.actions = []

        if action['type'] == ACTION_TYPE_CARD:
            self.actions.append({
                'player': player,
                'type': 'card',
                'move': {
                    'suit': action['move'].suit,
                    'rank': action['move'].rank,
                }
            })
        else: 
            raise NotImplementedError(f"action [{action['type']}] not implemented")

        self.step += 1

    def next_action(self):
        if self.step >= len(self.actions):
            return None

        action = self.actions[self.step]
        self.step += 1

        if action['type'] == 'card':
            return {
                'type': ACTION_TYPE_CARD,
                'move': Card(action['move'].suit, action['move'].rank)
            }
        
        raise SystemError(f"action [{action['type']}] not found")

    def save(self, path):
        data = {
            'next_player': self.next_player,
            'deck': {'hands': self.deck.hands},
            'trump': {'values': self.trump.values},
            'actions': self.actions
        }

        with open(path, 'w') as file:
            json.dump(data, file, indent=2)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        
        with open(path, 'r') as file:
            data = json.load(file)
            
            self.deck = Deck()
            self.deck.hands = data['deck']['hands']

            self.trump = Trump()
            self.trump.values = data['trump']['values']

            self.next_player = data['next_player']
            self.actions = data['actions']
            
            self.step = 0
            
        return self.reset_env()