from typing import Optional, Tuple, Iterable
from src.models.card import Card
from src.models.trump import Trump

class Collection(list[Card]):
    def __init__(self, cards: Iterable[Card] = None):
        super().__init__(cards or [])

    @property
    def cards(self) -> list[Card]:
        return list(self)

    def winner(self, trump: Trump) -> Tuple[Optional[Card], Optional[int]]:
        if not self:
            return None, None
            
        best_idx, best_card = 0, self[0]
        for i in range(1, len(self)):
            if self[i].beats(trump, best_card):
                best_idx, best_card = i, self[i]
        return best_card, best_idx
    
    def value(self, trump: Trump) -> int:
        return sum(card.value(trump) for card in self)

    def sort(self, trump: Trump) -> "Collection":
        # Calculate max value, sum, and count per suit
        suit_stats = {}
        for card in self:
            if card.suit not in suit_stats:
                suit_stats[card.suit] = {'max': 0, 'sum': 0, 'count': 0}
            suit_stats[card.suit]['max'] = max(suit_stats[card.suit]['max'], card.value(trump))
            suit_stats[card.suit]['sum'] += card.value(trump)
            suit_stats[card.suit]['count'] += 1
        
        # Sort: trump first, then by max value, sum, count (all descending), suit number, then card value within suit
        super().sort(key=lambda c: (
            c.suit != trump.suit,           # Trump cards first
            -suit_stats[c.suit]['max'],     # Sort suits by max value (highest first)
            -suit_stats[c.suit]['sum'],     # If equal, by sum (highest first)
            -suit_stats[c.suit]['count'],   # If equal, by count (most cards first)
            c.suit,                         # If still equal, by suit number (ascending)
            -c.value(trump),                # Within suit, highest value first
            -c.rank                         # Tiebreaker
        ))
        return self

        