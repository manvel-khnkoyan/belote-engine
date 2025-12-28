from copy import deepcopy
import os
from typing import List
import pickle
from .record import Record
from src.models.card import Card
from src.models.trump import Trump

class Result:
    def __init__(self, hands: List[List[Card]], trump: Trump, records: List[Record], scores: List[int] = None):
        self.hands = deepcopy(hands)
        self.trump = deepcopy(trump)
        self.records = records
        self.scores = scores if scores is not None else [0, 0]

    def save(self, path: str):
        # Remove existing file to avoid conflicts
        if os.path.exists(path):
            os.remove(path)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'Result':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        try:
            # Header lines
            header = [
                f"Result(trump={repr(self.trump)},",
            ]

            # Blank line between header and records for clarity
            header.append("")

            # Format records one-per-line, indenting multi-line record reprs
            record_lines = ["records:"]
            for r in self.records:
                rstr = repr(r)
                # indent each line of the record representation
                indented = "\n    ".join(rstr.splitlines())
                record_lines.append(f"  {indented}")

            footer = [")"]

            return "\n".join(header + record_lines + footer)
        except Exception:
            return "Result(...)"