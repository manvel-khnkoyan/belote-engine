
from abc import ABC
from typing import List
from src.models.set import Set

class Action(ABC):
    def __repr__(self):
        return self.__class__.__name__
    
    def __eq__(self, other):
        return self.__class__ == other.__class__

class ActionPass(Action):
    def __repr__(self):
        return "Pass"

class ActionPlayCard(Action):
    def __init__(self, card):
        self.card = card

    def __repr__(self):
        return f"PlayCard({self.card})"
    
    def __eq__(self, other):
        return isinstance(other, ActionPlayCard) and self.card == other.card

class ActionShowSet(Action):
    def __init__(self, set: Set):
        self.set = set

    def __repr__(self):
        return f"ShowSet({self.set})"
    
    def __eq__(self, other):
        return isinstance(other, ActionShowSet) and self.set == other.set

class ActionAnnounceBelote(Action):
    def __repr__(self):
        return "AnnounceBelote"
