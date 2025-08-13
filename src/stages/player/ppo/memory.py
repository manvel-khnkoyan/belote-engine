import time
import numpy as np


class PPOMemory:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.clear()
        self.seed = int(time.time() * 1000)

    def clear(self):
        self.action_types = []
        self.actions = []
        self.probabilities = []
        self.tables = []
        self.trumps = []
        self.values = []
        self.log_probs = []
        self.rewards = []

    def add_experience(self, experience):
        self.action_types.append(experience['action_type'])
        self.actions.append(experience['action'])
        self.probabilities.append(experience['probability'])
        self.tables.append(experience['table'])
        self.trumps.append(experience['trump'])
        self.values.append(experience['value'])
        self.log_probs.append(experience['log_prob'])
        self.rewards.append(0.0)

    def sample(self, indices):
        return {
            'action_types': [self.action_types[i] for i in indices],
            'actions': [self.actions[i] for i in indices],
            'probabilities': [self.probabilities[i] for i in indices],
            'tables': [self.tables[i] for i in indices],
            'trumps': [self.trumps[i] for i in indices],
            'values': [self.values[i] for i in indices],
            'log_probs': [self.log_probs[i] for i in indices],
            'rewards': [self.rewards[i] for i in indices]
        }

    def updated_last_rewards(self, reward_value, last_n=1, decay_factor=2):
        for i in range(last_n):
            idx = len(self.rewards) - 1 - i
            if idx >= 0:
                self.rewards[idx] += reward_value / (decay_factor ** i)

    def __len__(self):
        return len(self.actions)
    
    def __str__(self):
        return (f"PPOMemory(size={len(self)}, "
                f"last_action_type={self.action_types[-1] if self.action_types else None}, "
                f"last_action={self.actions[-1] if self.actions else None}, "
                f"last_reward={self.rewards[-1] if self.rewards else None})")