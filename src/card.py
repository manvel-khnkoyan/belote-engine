# Suit symbols for rendering
SUIT_SYMBOLS = ['♠', '♥', '♦', '♣']
RANK_SYMBOLS = [' 7', ' 8', ' 9', '10', ' J', ' Q', ' K', ' A']

class Card:
    def __init__(self, suit: int, rank: int):
        """
        Validate and initialize a card with a suit and rank.
        Args:
            suit: Suit index (0-3)
            rank: Rank index (0-7)
        Raises:
            ValueError: If suit or rank is out of bounds
        """
        if suit < 0 or suit > 3:
            raise ValueError(f"Invalid suit: {suit}. Must be between 0 and 3.")
        if rank < 0 or rank > 7:
            raise ValueError(f"Invalid rank: {rank}. Must be between 0 and 7.")

        self.suit = suit
        self.rank = rank
    
    def __repr__(self):
        return f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}"
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.suit, self.rank))
    
    def copy(self):
        return Card(self.suit, self.rank)
    
    def higher_than(self, other, trump=None):
        if self.suit == other.suit:
            self_value = self.value(trump)
            other_value = other.value(trump)

            if self_value == other_value:
                return self.rank > other.rank

            return self_value > other_value
        
        if self.is_trump(trump):
            return True
        
        return False
    
    def change_suit(self, transform):
        self.suit = transform(self.suit)

        return self
    
    def is_trump(self, trump):
        return trump.values[self.suit] == 1
    
    def value(self, trump):
        # Define values dictionary based on rank indices
        values = {
            7: 11,  # ACE
            3: 10,  # TEN
            6: 4,   # KING
            5: 3,   # QUEEN
            4: 2,   # JACK
            2: 0,   # NINE
            1: 0,   # EIGHT
            0: 0    # SEVEN
        }

        if trump.has_trump() == False:
            values[7] = 19  # ACE becomes highest
            return values[self.rank]

        # Adjust values for trump cards
        if self.is_trump(trump):
            values[4] = 20  # JACK becomes highest in trump
            values[2] = 14  # NINE becomes second highest in trump
            return values[self.rank]

        return values[self.rank]
    
    def __getstate__(self):
        return {"suit": self.suit, "rank": self.rank}
    
    def __setstate__(self, state):
        self.suit = state["suit"]
        self.rank = state["rank"]