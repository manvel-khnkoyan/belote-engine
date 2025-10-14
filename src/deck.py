import random
from src.card import Card
from src.trump import Trump

class Deck:
    def __init__(self):
        self.hands = [[], [], [], []]
        self.cards = []
        self.reset()

    def __getitem__(self, player):
        if player < 0 or player > 3:
            raise ValueError(f"Invalid player index: {player}. Must be between 0 and 3.")
        return self.hands[player]
    
    @staticmethod
    def new_cards():
        cards = []
        for rank in range(8):
            for suit in range(4):
                cards.append(Card(suit, rank))
        return cards

    def reset(self):
        self.cards = self.new_cards()
        random.shuffle(self.cards)

    def deal_cards(self, num_cards_per_player: int):
        for _ in range(num_cards_per_player):
            for player in range(4):
                self.hands[player].append(self.cards.pop())
    
    def remove_card(self, player_idx, card: Card):
        # player_hand is directly the list of cards
        player_hand = self[player_idx]  # This is already the list of cards
        
        # Find the exact card object (not just one with matching suit/rank)
        for i, c in enumerate(player_hand):
            if c == card:
                player_hand.pop(i)
                return
        
        # If we get here, the card wasn't found
        raise ValueError(f"Card {card} not found in player {player_idx}'s hand")
    
    def no_cards_left(self) -> bool:
        for hand in self.hands:
            if len(hand) > 0:
                return False

        return True
        
    def sort_hands(self, trump: Trump):
        for player in range(4):
            player_hand = self.hands[player]

            # group cards by suits
            grouped_cards = {suit: [] for suit in range(4)}
            for card in player_hand:
                grouped_cards[card.suit].append(card)

            # Sort groups by value sum descending
            sorted_groups = sorted(grouped_cards.values(), key=lambda group: sum(card.value(trump) for card in group), reverse=True)

            # Sort by trump suit
            if trump.mode == Trump.REGULAR:
                sorted_groups.sort(key=lambda group: 0 if len(group) > 0 and group[0].suit == trump.suit else 1)

            self.hands[player] = [card for group in sorted_groups for card in group]

    def copy(self):
        new_deck = Deck()
        new_deck.cards = self.cards.copy()
        for i in range(4):
            new_deck.hands[i] = self.hands[i].copy()

        return new_deck
    
    def __getstate__(self):
        return {
            "hands": self.hands,
            "cards": self.cards
        }
    
    def __setstate__(self, state):
        self.hands = state["hands"]
        self.cards = state["cards"]