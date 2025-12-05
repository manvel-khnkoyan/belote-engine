from typing import List, Any
import pickle
import json
from .record import Record
from src.models.card import Card
from src.models.trump import Trump

class Result:
    def __init__(self, hands: List[List[Card]], trump: Trump, records: List[Record]):
        self.hands = hands
        self.trump = trump
        self.records = records
        self.scores = [0, 0]  # Team scores

    def save(self, path: str):
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