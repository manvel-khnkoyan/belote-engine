from src.card import Card

class ActionPlayCard():
    def __init__(self, card: Card):
        self.card = card

    def __getstate__(self):
        return self.card
    
    def __setstate__(self, state):
        self.card = state

    def __repr__(self):
        return f"Play({self.card})"

