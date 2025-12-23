from typing import Optional, Tuple, Iterable
from src.models.card import Card
from src.models.trump import Trump, TrumpMode

class Canonical():
        
    @staticmethod
    def create_transform_map(cards: Iterable[Card], trump: Trump) -> Tuple[dict[int, int], dict[int, int]]:
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
        if trump.mode == TrumpMode.Regular:
            # If trump is regular, ensure trump suit is first
            trump_suit = trump.suit
            ordered = [trump_suit] + sorted([s for s in range(4) if s != trump_suit], 
                                           key=lambda s: (-strengths[s], s))
        else:
            ordered = sorted(range(4), key=lambda s: (-strengths[s], s))
        
        can_map = {orig: canon for canon, orig in enumerate(ordered)}
        reverse = {canon: orig for orig, canon in can_map.items()}

        return can_map, reverse