from const import Suits

# Trump modes
TrumpRegularMode = 1
TrumpNoTrumpMode = 2

class Trump:
    def __init__(self, mode: int, suit: int | None):
        assert mode in [1, 2], f"Invalid mode: {mode}"
        assert suit is None or 0 <= suit <= 3, f"Invalid suit: {suit}"
        self.mode = mode
        self.suit = suit
    
    def __iter__(self):
        return iter((self.mode, self.suit))
    
    def __repr__(self):
        if self.mode == TrumpNoTrumpMode:
            return "No Trump"
        if self.mode == TrumpRegularMode:
            return f"Trump {Suits[self.suit]}"

