from typing import Optional, Tuple, Iterable
from src.models.card import Card
from src.models.trump import Trump

class Cards():

    @staticmethod
    def winner(cards: list[Card], trump: Trump) -> Tuple[Optional[Card], Optional[int]]:
        if not cards:
            return None, None
            
        best_idx, best_card = 0, cards[0]
        for i in range(1, len(cards)):
            if cards[i].beats(trump, best_card):
                best_idx, best_card = i, cards[i]
        return best_card, best_idx
    
    @staticmethod
    def value(cards: list[Card], trump: Trump) -> int:
        return sum(card.value(trump) for card in cards)

    @staticmethod
    def sort(cards: list[Card], trump: Trump) -> list[Card]:
        # Calculate max value, sum, and count per suit
        suit_stats = {}
        for card in cards:
            if card.suit not in suit_stats:
                suit_stats[card.suit] = {'max': 0, 'sum': 0, 'count': 0}
            suit_stats[card.suit]['max'] = max(suit_stats[card.suit]['max'], card.value(trump))
            suit_stats[card.suit]['sum'] += card.value(trump)
            suit_stats[card.suit]['count'] += 1
        
        # Sort: trump first, then by max value, sum, count (all descending), suit number, then card value within suit
        cards.sort(key=lambda c: (
            c.suit != trump.suit,           # Trump cards first
            -suit_stats[c.suit]['max'],     # Sort suits by max value (highest first)
            -suit_stats[c.suit]['sum'],     # If equal, by sum (highest first)
            -suit_stats[c.suit]['count'],   # If equal, by count (most cards first)
            c.suit,                         # If still equal, by suit number (ascending)
            -c.value(trump),                # Within suit, highest value first
            -c.rank                         # Tiebreaker
        ))

        return cards
        
    @staticmethod
    def suit_canonical_map(cards: list[Card]) -> dict[int, int]:
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