from .actions import Action, ActionPlayCard, ActionPass
from src.models.card import Card
from src.models.trump import Trump
from typing import List

"""
State is personal from first person view.
"""
class State:
    def __init__(self, cards: List[Card], trump: Trump):
        self.cards = cards
        self.trump = trump
        self.round = 0
        self.table = []
        # self.declared_action_types = []

    def observe(self, player: int, action: Action):
        if isinstance(action, ActionPlayCard):
            self.table.append(action.card)
            if player == 0:
                self.cards.remove(action.card)
            
            if len(self.table) == 4:
                self.round += 1
                self.table = []

    def __repr__(self):
        return f"Round={self.round} State(cards={self.cards}, trump={self.trump}, table={self.table})"