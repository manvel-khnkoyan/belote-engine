import random
from typing import Callable, Tuple
from card import Card

class Deck:
    def __init__(self):
        self.cards: list[Card] = []
        for _ in range(4):  # 4 suits
            for rank in range(8):  # 8 ranks (7 to Ace)
                self.cards.append(Card(suit=_, rank=rank))
        
        random.shuffle(self.cards)

    def deal(self) -> list[list[Card]]:
        hands: list[list[Card]] = [[] for _ in range(4)]
        for i in range(4):
            hands[i] = self.cards[i*8:(i+1)*8]
        return hands

    """ Creates a canonical mapping for suits based on their distribution in the deck.
        Returns a tuple of (transform_function, reverse_function).  """
    def transform_suits_canonical(self) -> Tuple[Callable[[int], int], Callable[[int], int]]:
        hands = self.deal()
        num_suits = 4
        
        # Calculate strengths
        strengths = [0.0] * num_suits
        for hand in hands:
            counts, ranks = [0] * 4, [0] * 4
            for c in hand:
                counts[c.suit] += 1
                ranks[c.suit] += c.rank
            for s in range(4):
                if counts[s]:
                    strengths[s] += counts[s] * 10 + ranks[s] / counts[s]
        
        # Create mappings
        ordered = sorted(range(4), key=lambda s: (-strengths[s], s))
        suit_map = {orig: canon for canon, orig in enumerate(ordered)}
        rev_map = {canon: orig for orig, canon in suit_map.items()}
        
        # Update cards
        for card in self.cards:
            card.suit = suit_map[card.suit]
        
        return (lambda s: suit_map[s]), (lambda c: rev_map[c])