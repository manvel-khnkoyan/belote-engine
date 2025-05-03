import numpy as np
import torch
import src.card as Card

class Table:
    """
    Represents the cards currently on the table in a Belote game.
    
    Tracks up to 4 cards visible to the current player (the ones played by other players).
    Each card is represented by Card objects.
    """
    
    def __init__(self):
        self.max_cards = 4
        self.num_cards = 0
        
        # Initialize with empty slots (None indicates no card)
        self.cards = np.array([None, None, None, None], dtype=object)

    def __str__(self):
        return f"Table: {self.cards[:self.num_cards]}"
    
    def __repr__(self):
        return f"Table(cards={self.cards[:self.num_cards]})"
    
    def __len__(self):
        return self.num_cards
    
    def __getitem__(self, player):
        if player < 0 or player >= self.max_cards:
            raise IndexError("Player index out of range")
        
        return self.cards[player]

    def clear(self):
        self.cards = np.array([None, None, None, None], dtype=object)
        self.num_cards = 0
    
    def add(self, card):
        if self.num_cards >= self.max_cards:
            return False
        
        # Add the card
        self.cards[self.num_cards] = card
        self.num_cards += 1
        
        return True
    
    def is_full(self):
        return self.num_cards >= self.max_cards
    
    def is_empty(self):
        return self.num_cards == 0
    
    def total_points(self, trump=None):
        total_points = 0
        for card in self.cards[:self.num_cards]:
            if card is not None:
                total_points += card.value(trump)
        
        return total_points
    
    def winner_card(self, trump=None):
        winner_card = self.cards[0] if self.num_cards > 0 else None
        winner_index = 0

        for index, card in enumerate(self.cards[:self.num_cards]):
            if card is not None and winner_card is not None:
                if card.higher_than(winner_card, trump):  # Changed from higher_then to higher_than
                    winner_card = card
                    winner_index = index
        
        return winner_card, winner_index

    def to_tensor(self):
        # Create a tensor with 3 positions, each holding an 8×4 grid (rank × suit)
        # This allows each card to be processed by a 2D convolutional network
        tensor = torch.zeros(3, 8, 4)
        
        # Get the last 3 cards placed on the table (or fewer if there are fewer)
        start_idx = max(0, self.num_cards - 3)
        for i in range(start_idx, self.num_cards):
            tensor_idx = i - start_idx
            card = self.cards[i]
            if card is not None:
                # Set 1.0 at the position corresponding to this card's rank and suit
                # This creates a one-hot encoding in a 2D grid format
                tensor[tensor_idx, card.rank, card.suit] = 1.0
        
        return tensor
        
    def change_suits(self, transform):
        for i in range(self.num_cards):
            if self.cards[i] is not None:
                self.cards[i] = self.cards[i].change_suit(transform)
        
        return self
        
    def copy(self):
        new = Table()
        new.cards = np.copy(self.cards)

        return new

