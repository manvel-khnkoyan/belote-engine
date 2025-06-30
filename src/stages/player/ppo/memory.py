import time
import numpy as np


class PPOMemory:
    def __init__(self):
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
        self.rewards.append(experience['reward'])


    def cut_experience(self, keep_n_from_end):
        if keep_n_from_end <= 0 or len(self.actions) == 0:
            return
        
        # Calculate the index to keep
        start_index = max(0, len(self.actions) - keep_n_from_end)

        # Slice the memory lists to keep only the last `keep_n_from_end` entries
        self.action_types = self.action_types[start_index:]
        self.actions = self.actions[start_index:]
        self.probabilities = self.probabilities[start_index:]
        self.tables = self.tables[start_index:]
        self.trumps = self.trumps[start_index:]
        self.values = self.values[start_index:]
        self.log_probs = self.log_probs[start_index:]
        self.rewards = self.rewards[start_index:]

    def random_batch(self, batch_size=64):
        if len(self.actions) == 0:
            return None
        
        # Increment the seed to ensure randomness in sampling
        self.seed += 1

        # Handle case where batch_size > available actions
        actual_batch_size = min(batch_size, len(self.actions))
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(self.actions), size=actual_batch_size, replace=False)

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



import time
import numpy as np


class PPOMemory:
    def __init__(self):
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
        """Add experience to memory with proper reward initialization"""
        self.action_types.append(experience['action_type'])
        self.actions.append(experience['action'])
        self.probabilities.append(experience['probability'])
        self.tables.append(experience['table'])
        self.trumps.append(experience['trump'])
        self.values.append(experience['value'])
        self.log_probs.append(experience['log_prob'])
        # Fix: Initialize reward to 0.0 instead of accessing non-existent key
        self.rewards.append(0.0)

    def cut_experience(self, keep_n_from_end):
        """Keep only the last N experiences"""
        if keep_n_from_end <= 0 or len(self.actions) == 0:
            return
        
        # Calculate the index to keep
        start_index = max(0, len(self.actions) - keep_n_from_end)

        # Slice the memory lists to keep only the last `keep_n_from_end` entries
        self.action_types = self.action_types[start_index:]
        self.actions = self.actions[start_index:]
        self.probabilities = self.probabilities[start_index:]
        self.tables = self.tables[start_index:]
        self.trumps = self.trumps[start_index:]
        self.values = self.values[start_index:]
        self.log_probs = self.log_probs[start_index:]
        self.rewards = self.rewards[start_index:]

    def random_batch(self, batch_size=64):
        """Get a random batch of experiences for training"""
        if len(self.actions) == 0:
            return None
        
        # Increment the seed to ensure randomness in sampling
        self.seed += 1

        # Handle case where batch_size > available actions
        actual_batch_size = min(batch_size, len(self.actions))
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(self.actions), size=actual_batch_size, replace=False)

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
        """Return the number of experiences stored"""
        return len(self.actions)
    
    def is_empty(self):
        """Check if memory is empty"""
        return len(self.actions) == 0