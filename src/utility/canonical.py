from typing import Optional, Tuple, Iterable
from src.models.card import Card
from src.models.trump import Trump

class Canonical():
        
    @staticmethod
    def create_transform_map(cards: Iterable[Card]) -> dict[int, int]:
        num_suits = 4
        
        # Calculate strengths
        strengths = [0.0] * num_suits
        counts = [0] * num_suits
        for card in cards:
            counts[card.suit] += 1
            strengths[card.suit] += card.rank
        
        for s in range(num_suits):
            if counts[s]:
                strengths[s] /= counts[s]
        
        # Create mappings
        ordered = sorted(range(4), key=lambda s: (-strengths[s], s))
        can_map = {orig: canon for canon, orig in enumerate(ordered)}

        return can_map