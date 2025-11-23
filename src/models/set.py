from collections import defaultdict
from functools import cmp_to_key
from typing import List

from card import Card
from trump import Trump, TrumpMode

class Set:
    Tierce = 1
    Quarte = 2
    Quinte = 3
    Quartet = 4

    def __init__(self, type: int, cards: List[Card]):
        self.type = type
        self.cards = cards

    def value(self, trump: Trump) -> int:
        if self.type == Set.Quinte: return 100
        if self.type == Set.Quarte: return 50
        if self.type == Set.Tierce: return 20
        if self.type == Set.Quartet:
            # Rank: 7=0, 8=1, 9=2, 10=3, J=4, Q=5, K=6, A=7
            rank = self.cards[0].rank
            if rank == 2: return 0 if trump.mode == TrumpMode.NoTrump else 140 # 9
            if rank == 4: return 100 if trump.mode == TrumpMode.NoTrump else 200 # J
            if rank == 7: return 190 if trump.mode == TrumpMode.NoTrump else 100 # A
            if rank in (3, 5, 6): return 100 # 10, Q, K
            return 0 # 7, 8
        return 0

    def beats(self, trump: Trump, other: "Set") -> bool:
        # Quartet priority
        if (self.type == Set.Quartet) != (other.type == Set.Quartet):
            return self.type == Set.Quartet
        
        if self.type == Set.Quartet:
            return self.value(trump) > other.value(trump)

        # Sequence priority
        is_trump = trump.mode == TrumpMode.Regular and self.cards[0].suit == trump.suit
        other_trump = trump.mode == TrumpMode.Regular and other.cards[0].suit == trump.suit
        
        if is_trump != other_trump:
            return is_trump
            
        if self.type != other.type:
            return self.type > other.type
            
        return self.cards[-1].rank > other.cards[-1].rank

    @staticmethod
    def extract(cards: List[Card]) -> List["Set"]:
        sets = []
        
        # Sequences
        by_suit = defaultdict(list)
        for c in cards: by_suit[c.suit].append(c)
        
        for suit_cards in by_suit.values():
            if len(suit_cards) < 3: continue
            suit_cards.sort(key=lambda c: c.rank)
            
            seq = [suit_cards[0]]
            for i in range(1, len(suit_cards)):
                if suit_cards[i].rank == seq[-1].rank + 1:
                    seq.append(suit_cards[i])
                else:
                    if len(seq) >= 3: sets.append(Set._from_seq(seq))
                    seq = [suit_cards[i]]
            if len(seq) >= 3: sets.append(Set._from_seq(seq))

        # Quartets
        by_rank = defaultdict(list)
        for c in cards: by_rank[c.rank].append(c)
        sets.extend(Set(Set.Quartet, cs) for cs in by_rank.values() if len(cs) == 4)
        
        return sets

    @staticmethod
    def _from_seq(cards: List[Card]) -> "Set":
        t = {3: Set.Tierce, 4: Set.Quarte}.get(len(cards), Set.Quinte)
        return Set(t, cards)

    @staticmethod
    def sort(sets: List["Set"], trump: Trump) -> List["Set"]:
        return sorted(sets, key=cmp_to_key(lambda a, b: 1 if a.beats(trump, b) else -1 if b.beats(trump, a) else 0), reverse=True)