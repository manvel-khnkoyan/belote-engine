
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

class ActionDeclareSets(Action):
    def __init__(self, sets: List[Set]):
        self.sets = sets

    def __repr__(self):
        return f"DeclareSets({self.sets})"
    
    def __eq__(self, other):
        return isinstance(other, ActionDeclareSets) and self.sets == other.sets

class ActionDeclareBets(Action):
    def __repr__(self):
        return "DeclareBets"
