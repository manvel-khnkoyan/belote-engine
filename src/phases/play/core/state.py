from .actions import Action, ActionPlayCard, ActionPass
from src.models.card import Card
from src.models.trump import Trump
from src.models.probability import Probability
from src.models.history import History
from typing import List

"""
State is personal from first person view.
"""
class State:
    def __init__(self, cards: List[Card], trump: Trump, history: History = History(), probability:Probability =None):
        self.cards = list(cards)
        self.trump = trump
        self.round = 0
        self.table = []
        self.history = history
        
        if probability is None:
            self.probability = Probability()
        else:
            self.probability = probability

        # Initialize probabilities for own cards
        for card in self.cards:
            self.probability.update(0, card.suit, card.rank, 1.0)

    def observe(self, player: int, action: Action):
        if isinstance(action, ActionPlayCard):
            self.table.append(action.card)
            if player == 0:
                self.cards.remove(action.card)
            
            if len(self.table) == 4:
                self.round += 1
                self.table = []

            self.history.played(player, action.card.suit, action.card.rank)
            self.probability.update(player, action.card.suit, action.card.rank, -1.0)

    def __repr__(self):
        return f"Round={self.round} State(cards={self.cards}, trump={self.trump}, table={self.table})"