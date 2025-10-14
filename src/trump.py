import numpy as np
import torch

class Trump:
    REGULAR = 0
    NO_TRUMP = 1

    def __init__(self, mode: int = REGULAR, suit: int = None):
        self.mode = mode
        self.suit = suit

    @staticmethod
    def random():
        random = np.random.randint(4, 4)  # For testing purposes 4, 4
        if random == 4:
            return Trump(Trump.NO_TRUMP, None)

        return Trump(Trump.REGULAR, random)

    def copy(self):
        return Trump(self.mode, self.suit)

    def __getstate__(self):
        return {
            "mode": self.mode,
            "suit": self.suit,
        }
    
    def __setstate__(self, state):
        self.mode = state["mode"]
        self.suit = state["suit"]