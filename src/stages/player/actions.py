from src.card import Card

class Action:
    TYPE_PLAY = 0
    TYPE_BELOTE = 1
    TYPE_REBELOTE = 2

class ActionCardMove(Action):
    def __init__(self, card: Card):
        self.type = Action.TYPE_PLAY
        self.card = card
    
    def __eq__(self, action):
        return self.type == action.type and self.card == action.card
    
    def __getstate__(self):
        return {
            "type": self.type,
            "card": self.card
        }
    
    def __setstate__(self, state):
        self.type = state["type"]
        self.card = state["card"]

