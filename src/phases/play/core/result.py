from typing import List
from record import Record
from src.models.deck import Deck
from src.models.trump import Trump

class Result:
    def __init__(self, deck: Deck, trump: Trump, records: List[Record]):
        self.deck = deck
        self.trump = trump
        self.records = records