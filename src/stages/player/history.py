import os
import pickle
from src.stages.player.env import BeloteEnv

class History:
    def __init__(self):
        self.actions = []
        self.deck = None
        self.trump = None
        self.player = None
        self.reset()

    def reset(self):
        self.cursor = 0

    @staticmethod
    def create(trump, deck, next_player):
        h = History()
        h.trump = trump.copy()
        h.deck = deck.copy()
        h.player = next_player

        return h
    
    def record_action(self, player, action):
        self.actions.append({
            'player': player,
            'action': action,
        })

    def create_env(self):
        """Create a new environment for the history"""
        return BeloteEnv(trump=self.trump.copy(), deck=self.deck.copy(), next_player=self.player)

    def get_next_action(self):
        if self.cursor >= len(self.actions):
            return None
        
        item = self.actions[self.cursor]
        self.cursor += 1

        return item['action'], item['player']

    def save(self, path):
        with open(path, 'wb') as file:  # Note: binary mode
            pickle.dump(self.__getstate__(), file)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        
        self.reset()
        with open(path, 'rb') as file:  # Note: binary mode
            data = pickle.load(file)
            self.__setstate__(data)
            self.reset()

    def __getstate__(self):
        return {
            "trump": self.trump,
            "deck": self.deck,
            "next_player": self.player,
            "actions" : self.actions
        }

    def __setstate__(self, state):
        self.trump = state["trump"]
        self.deck = state["deck"]
        self.player = state["next_player"]
        self.actions = state["actions"]
