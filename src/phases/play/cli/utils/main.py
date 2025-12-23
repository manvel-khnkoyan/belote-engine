from src.utility.canonical import Canonical
from src.suits import Suits
from src.models.card import Card
from src.models.trump import Trump, TrumpMode

def transform_canonical(trump: Trump, hands: list[list[Card]]):
    # Create canonical mapping based on player 0's hand
    suit_map, _ = Canonical.create_transform_map(hands[0], trump)

    # Transform trump and hands using canonical mapping
    Suits.transform(suit_map)
    trump.suit = suit_map[trump.suit] if trump.suit is not None else None
    hands = [[Card(suit_map[card.suit], card.rank) for card in hand] for hand in hands]

    return trump, hands
