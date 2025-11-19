from typing import Optional, List
from src.tricks.actions.types import ActionTypes

class Action:
    """Base class for all game actions"""
    
    def __init__(self, type: int):
        self.type = type
    
    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.type})"


class ActionPlayCard(Action):
    """Action representing a player playing a card"""
    
    def __init__(self, card):
        super().__init__(ActionTypes.PLAY_CARD)
        self.card = card
    
    def __repr__(self):
        return f"ActionPlayCard({self.card})"
    
    def __getstate__(self):
        return {"card": self.card}
    
    def __setstate__(self, state):
        self.type = ActionTypes.PLAY_CARD
        self.card = state["card"]


class ActionSayBelote(Action):
    """Action representing a player announcing Belote/Rebelote"""
    
    def __init__(self):
        super().__init__(ActionTypes.SAY_BELOTE)
    
    def __repr__(self):
        return "ActionSayBelote()"
    
    def __getstate__(self):
        return {}
    
    def __setstate__(self, state):
        self.type = ActionTypes.SAY_BELOTE


class ActionAnnounceSequence(Action):
    """Action representing a player announcing a sequence (tierce, quarte, etc.)"""
    
    def __init__(self, cards: List):
        super().__init__(ActionTypes.ANNOUNCE_SEQUENCE)
        self.cards = cards
    
    def __repr__(self):
        return f"ActionAnnounceSequence({self.cards})"
    
    def __getstate__(self):
        return {"cards": self.cards}
    
    def __setstate__(self, state):
        self.type = ActionTypes.ANNOUNCE_SEQUENCE
        self.cards = state["cards"]