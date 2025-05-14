import os
import numpy as np
import pickle
from src.deck import Deck
from src.states.trump import Trump

class History:
    def __init__(self, env):
        self.env = env
        self.cursor = 0
        self.actions = []

    def __getstate__(self):
        return {
            "env": self.env,
            "actions": self.actions
        }

    def reset(self):
        self.cursor = 0

    def read(self):
        if self.cursor >= len(self.actions):
            return None
        
        item = self.actions[self.cursor]

        return item.action, item.player

    def write(self, player, action):
        self.actions.append({
            'player': player,
            'action': action,
        })

    def save(self, path):
        with open(path, 'wb') as file:  # Note: binary mode
            pickle.dump({
                "cursor": self.cursor,
                "actions": self.actions
                # Don't include env as it might be too complex
            }, file)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        
        with open(path, 'rb') as file:  # Note: binary mode
            data = pickle.load(file)
            self.cursor = data["cursor"]
            self.actions = data["actions"]

    def __getstate__(self):
        return {
            "env": self.env,
            "cursor": self.cursor,
            "actions": self.actions
        }

    def __setstate__(self, state):
        self.env = state["env"]
        self.cursor = state["cursor"]
        self.actions = state["actions"]
