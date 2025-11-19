import numpy as np

class Probability:
    """Tracks card probabilities: (4, 4, 8) matrix (player, suit, rank).
    Values: 1.0 (has), 0.0 (not), -1.0 (played), 0-1 (prob)."""

    def __init__(self, matrix=None):
        self.matrix = matrix if matrix is not None else np.full((4, 4, 8), 0.25, dtype=np.float32)

    def update(self, player, suit, rank, val):
        """Updates probability for a card, redistributing remaining probability to others."""
        if self.matrix[0, suit, rank] == -1.0: return False # Card played
        
        # If setting to 1.0 or -1.0 (certainty), clear others
        if abs(abs(val) - 1.0) < 1e-6:
            self.matrix[:, suit, rank] = 0.0
            self.matrix[player, suit, rank] = float(round(val))
            return True

        current = self.matrix[player, suit, rank]
        if abs(current - val) < 1e-6: return True

        # Redistribute remaining probability
        others_mask = np.arange(4) != player
        others = self.matrix[others_mask, suit, rank]
        others_sum = np.sum(np.abs(others))
        
        self.matrix[player, suit, rank] = val
        remaining = 1.0 - abs(val)
        
        if others_sum < 1e-6:
            self.matrix[others_mask, suit, rank] = remaining / 3.0
        else:
            # Maintain relative proportions
            self.matrix[others_mask, suit, rank] = others * (remaining / others_sum)
            
        return True
    
    def extract(self, player, suit, rank, pct):
        """Extracts probability from other players and adds to target player."""
        if self.matrix[0, suit, rank] == -1.0: return False
        
        others_mask = np.arange(4) != player
        others_sum = np.sum(self.matrix[others_mask, suit, rank])
        
        if others_sum < 1e-6: return False
        
        return self.update(player, suit, rank, self.matrix[player, suit, rank] + others_sum * pct)

    def save(self, path='probability_matrix.npy'): np.save(path, self.matrix)
    def load(self, path='probability_matrix.npy'): self.matrix = np.load(path)