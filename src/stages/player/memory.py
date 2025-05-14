import numpy as np
import time

class Memory:
    def __init__(self):
        self.probability_tensors = []   # Store probability tensors
        self.tables_tensors = []        # Store table tensors
        self.trump_tensors = []         # Store trump tensors
        self.action_types = []
        self.actions = []               # Network actions
        self.actions_masks = []
        self.values = []
        self.rewards = []
        self.log_probs = []

        self.seed = int(time.time()) % 1000

    def __len__(self):
        return len(self.rewards)
        
    def store(self, probability_tensor, table_tensor, trump_tensor, action_type, action, actions_mask, value, reward, log_prob=None):
        self.probability_tensors.append(probability_tensor)
        self.tables_tensors.append(table_tensor)
        self.trump_tensors.append(trump_tensor)
        self.action_types.append(action_type)
        self.actions.append(action)
        self.actions_masks.append(actions_mask)
        self.values.append(value)
        self.rewards.append(reward)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        
    def clear(self):
        self.probability_tensors = []
        self.tables_tensors = []
        self.trump_tensors = []
        self.action_types = []
        self.actions = []
        self.actions_masks = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        
    def random_batches(self, batch_size):
        batch_start = np.arange(0, len(self.rewards), batch_size)
        indices = np.arange(len(self.rewards), dtype=np.int64)

        self.seed += 1 
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        
        batches = [indices[i:i+batch_size] for i in batch_start]

        return batches
    
    def update_last_reward(self, reward):
        if len(self.rewards) > 0:
            self.rewards[-1] = reward