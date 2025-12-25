import numpy as np

class History:
    """
    Tracks played cards history: (4, 4, 8) matrix (player, suit, rank).
    Values: 
        1.0: Card was played by this player
        0.0: Card was not played by this player
    """

    def __init__(self, matrix=None):
        self.matrix = matrix if matrix is not None else np.zeros((4, 4, 8), dtype=np.float32)

    def played(self, player: int, suit: int, rank: int) -> bool:
        """
        Marks a card as played by a specific player.
        Returns True if update was successful, False if card was already played.
        """
        # Check if card is already played by anyone
        if np.any(self.matrix[:, suit, rank] > 0.5):
            return False
            
        self.matrix[player, suit, rank] = 1.0
        return True


    def save(self, path='history_matrix.npy'): 
        np.save(path, self.matrix)
