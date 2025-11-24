from .actions import Action
from src.models.card import Card
from src.models.trump import Trump
from typing import List

"""
State is personal from first person view.
"""
class State:
    def __init__(self, player_index: int, cards: List[Card], trump: Trump):
        self.player_index = player_index
        self.cards = cards
        self.trump = trump
        self.table = []
        self.declared_action_types = []

    def observe(self, player: int, action: Action):
        pass

    def __repr__(self):
        return f"State(cards={self.cards}, trump={self.trump}, table={self.table})"