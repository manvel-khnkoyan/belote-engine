from src.card import Card
import random

class Deck:
    def __init__(self):
        self.hands = [[], [], [], []]
        self.cards = []
        self.reset()

    def __getitem__(self, player):
        if player < 0 or player > 3:
            raise ValueError(f"Invalid player index: {player}. Must be between 0 and 3.")
        return self.hands[player]

    def reset(self):
        self.cards = []
        for rank in range(8):  # 0-7 for ranks
            for suit in range(4):  # 0-3 for suits
                self.cards.append(Card(suit, rank))
        random.shuffle(self.cards)

    def deal_cards(self, num_cards_per_player: int):
        for _ in range(num_cards_per_player):
            for player in range(4):
                self.hands[player].append(self.cards.pop())
    
    def remove_card(self, player_idx, card):
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
        
    def reorder_hands(self, trump):
        for player in range(4):
            player_hand = self.hands[player]
            
            # Define sorting key function
            def sort_key(card):
                # Primary sort: non-trump suits after trump suit (1 for non-trump, 0 for trump)
                is_not_trump = 0 if card.is_trump(trump) else 1
                
                # Secondary sort: by suit (only matters for non-trump suits)
                suit = card.suit
                
                # Tertiary sort: by value in DESCENDING order (negative so higher values come first)
                # We use the card's value method which accounts for trump status
                value = -card.value(trump)  # Negative to sort in descending order
                
                return (is_not_trump, suit, value)
            
            # Sort the player's hand using the custom sorting key
            self.hands[player] = sorted(player_hand, key=sort_key)

    def copy(self):
        new_deck = Deck()
        new_deck.cards = self.cards.copy()
        for i in range(4):
            new_deck.hands[i] = self.hands[i].copy()

        return new_deck