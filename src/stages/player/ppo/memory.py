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
        
        # Auto-trim if too large
        if len(self.actions) > self.max_size:
            self.cut_experience(self.max_size // 2)

    def cut_experience(self, keep_n_from_end):
        if keep_n_from_end <= 0 or len(self.actions) == 0:
            return
        
        start_index = max(0, len(self.actions) - keep_n_from_end)
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
        
        self.seed += 1
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
        return len(self.actions)
    
    def __str__(self):
        return (f"PPOMemory(size={len(self)}, "
                f"last_action_type={self.action_types[-1] if self.action_types else None}, "
                f"last_action={self.actions[-1] if self.actions else None}, "
                f"last_reward={self.rewards[-1] if self.rewards else None})")