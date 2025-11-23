
from abc import ABC
from typing import List
from src.models.set import Set

class Action(ABC):
    def __repr__(self):
        return self.__class__.__name__

class ActionPass(Action):
    def __repr__(self):
        return "Pass"

class ActionPlayCard(Action):
    def __init__(self, card):
        self.card = card

    def __repr__(self):
        return f"PlayCard({self.card})"
    
class ActionDeclareSetsYes(Action):
    def __init__(self, sets: List[Set]):
        self.sets = sets

    def __repr__(self):
        return f"DeclareSets[Yes]({self.sets})"
    
class ActionDeclareSetsNo(Action):
    def __repr__(self):
        return "DeclareSets[No]"

class ActionDeclareSetsNo(Action):
    def __repr__(self):
        return "DeclareSets[No]"
    
class ActionDeclareBetsYes(Action):
    def __repr__(self):
        return "DeclareBets[Yes]"
    
class ActionDeclareBetsNo(Action):
    def __repr__(self):
        return "DeclareBets[No]"