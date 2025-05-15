import os
import pickle
from src.stages.player.env import BeloteEnv

class History:
    def __init__(self, env=None):
        self.cursor = 0
        self.actions = []
        
        # Save Env :(
        if env != None:
            self.save_env(env)

    def reset(self):
        self.cursor = 0

    def save_env(self, env):
        self.env = {
            'deck': env.deck.copy(),
            'trump': env.trump.copy(),
            'player': env.next_player
        }

    def load_env(self):
        if self.env == None:
            raise ValueError("History snapshot not found, try to save env whle implementing history")

        return BeloteEnv(trump=self.env['trump'], deck=self.env['deck'], next_player=self.env['player'])

    def next_action(self):
        if self.cursor >= len(self.actions):
            return None
        
        item = self.actions[self.cursor]
        self.cursor += 1

        return item['action'], item['player']

    def add_action(self, player, action):
        self.actions.append({
            'player': player,
            'action': action,
        })

    def save(self, path):
        with open(path, 'wb') as file:  # Note: binary mode
            pickle.dump({
                "env": self.env,
                "actions": self.actions
            }, file)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        
        with open(path, 'rb') as file:  # Note: binary mode
            data = pickle.load(file)
            self.env = data["env"]
            self.actions = data["actions"]

    def __getstate__(self):
        return {
            "env": self.env,
            "actions" : self.actions
        }

    def __setstate__(self, state):
        self.env = state["env"]
        self.actions = state["actions"]
