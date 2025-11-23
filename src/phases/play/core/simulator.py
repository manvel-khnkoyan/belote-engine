from typing import List
from src.models.deck import Deck
from src.models.trump import Trump
from rules import Rules
from result import Result
from agent import Agent

class  Simulator():
    def __init__(self, rules: Rules):
        self.rules = rules

    def simulate(self, agents: List[Agent], deck: Deck, trump: Trump) -> Result:
        pass
