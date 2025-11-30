from src.const import Suits
from enum import IntEnum

class TrumpMode(IntEnum):
    Regular = 1
    NoTrump = 2
    AllTrump = 3

class Trump:
    def __init__(self, mode: int, suit: int | None):
        assert mode in [TrumpMode.Regular, TrumpMode.NoTrump, TrumpMode.AllTrump], f"Invalid mode: {mode}"
        assert suit is None or 0 <= suit <= 3, f"Invalid suit: {suit}"
        self.mode = mode
        self.suit = suit
    
    def __iter__(self):
        return iter((self.mode, self.suit))
    
    def __repr__(self):
        if self.mode == TrumpMode.NoTrump:
            return "No Trump"
        if self.mode == TrumpMode.AllTrump:
            return "All Trump"
        if self.mode == TrumpMode.Regular:
            return f"Trump {Suits[self.suit]}"

