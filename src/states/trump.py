import numpy as np
import torch
class Trump:
    """
    Represents the trump suit state in a Belote game.
    
    Tracks which suit is the trump suit, or if there is no trump.
    Trump is represented as a one-hot encoding for the 4 suits [Clubs, Diamonds, Hearts, Spades],
    with [0,0,0,0] indicating no trump.
    """
    
    def __init__(self):
        # Initialize with no trump (all zeros)
        self.values = np.zeros(4, dtype=np.int32)
    
    def clear(self):
        self.values = np.zeros(4, dtype=np.int32)

    def has_trump(self):
        return np.any(self.values > 0)
    
    def set_trump(self, values):
        if len(values) != 4:
            raise ValueError(f"Invalid trump values: {values}. Must be an array of length 4.")
        
        # Validate one-hot encoding (all 0s except for at most one 1)
        if not all(v in [0, 1] for v in values) or sum(values) > 1:
            raise ValueError(f"Invalid trump values: {values}. Must contain only 0s and 1s with at most one 1.")
        
        # Set the trump values
        self.values = np.array(values, dtype=np.int32)
    
    def set_random_trump(self, allow_no_trump=False):
        self.clear()

        # Randomly select a suit index (0-3)
        rng = np.random.default_rng(None)
        suit_index = int(rng.integers(0, 4 + allow_no_trump * 1))

        if suit_index == 5:
            return

        # Set the selected suit as trump
        self.values[suit_index] = 1
    
    def to_tensor(self):
        return torch.tensor(self.values, dtype=torch.float)
    
    def change_suits(self, transform):
        self.values = [self.values[transform(i)] for i in range(4)]

        return self
        
    def copy(self):
        new = Trump()
        new.values = np.copy(self.values)

        return new