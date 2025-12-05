import random
from src.models.card import Card

class Deck:
    
    @staticmethod
    def create(shuffle: bool = True) -> list[list[Card]]:
        cards: list[Card] = []
        for _ in range(4):  # 4 suits
            for rank in range(8):  # 8 ranks (7 to Ace)
                cards.append(Card(suit=_, rank=rank))
        
        if shuffle:
            random.shuffle(cards)

        return [cards[i:i+8] for i in range(0, 32, 8)]
