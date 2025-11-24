from typing import List
import pickle
from .record import Record
from src.models.deck import Deck
from src.models.trump import Trump

class Result:
    def __init__(self, deck: Deck, trump: Trump, records: List[Record]):
        self.deck = deck
        self.trump = trump
        self.records = records

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'Result':
        with open(path, 'rb') as f:
            return pickle.load(f)