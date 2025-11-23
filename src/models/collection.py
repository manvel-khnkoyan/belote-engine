from typing import Optional, Tuple, Iterable
from card import Card
from trump import Trump

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

    def sort(self, trump: Trump) -> "Collection":
        super().sort(key=lambda c: (c.suit != trump.suit, c.suit, -c.value(trump), -c.rank))
        return self

        