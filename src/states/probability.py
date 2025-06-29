import numpy as np
import torch

class Probability:
    """
    Represents the probability matrix state for Belote game.
    Tracks the probability of each player having each card.
    
    This states shows only probabilities that players have cards, or played cards.
    and not the winner of the trick.

    State dimensions: (4, 4, 8) - (player, suits, ranks)
    Values range from 0 to 1, where:
    - 1.0: Player definitely has this card
    - 0.0: Player definitely does not have this card
    - -1.0: Card has been played // possibly we can make memory lose by giving 0 to -1 ratio
    - Values between 0-1: Probability that player has this card
    """
    def __init__(self):
        self.matrix = np.full((4, 4, 8), 0.25, dtype=np.float32)

    def update(self, player, suit, rank, absolute_value):
        """
        Update a player's probability for a card to an absolute value,
        redistributing the difference proportionally among other players.
        Handles negative probabilities while ensuring the sum of absolute values is 1.0.
        
        Args:
            player: Player index (0-3)
            suit: Suit index (0-3)
            rank: Rank index (0-7)
            absolute_value: New probability value for target player
            
        Returns:
            success: Whether the update was successful
        """
        # Validate indices
        self._validate_indices(player, suit, rank)
        
        # Check if card has been played (-1.0)
        if any(self.matrix[p, suit, rank] == -1.0 for p in range(4)):
            return False  # Can't modify probabilities of played cards
        
        # If absolute value is 1.0 or -1.0, set all others to 0.0
        if abs(abs(absolute_value) - 1.0) < 1e-6:
            for p in range(4):
                if p != player:
                    self.matrix[p, suit, rank] = 0.0
            self.matrix[player, suit, rank] = float(np.round(absolute_value))
            return True
        
        # Get current values
        current_value = self.matrix[player, suit, rank]
        
        # Calculate difference that needs to be redistributed
        difference = current_value - absolute_value
        
        # If no difference, no need to update
        if abs(difference) < 1e-6:
            return True
        
        # Calculate sum of absolute probabilities of other players
        abs_others_sum = 0.0
        for p in range(4):
            if p != player:
                abs_others_sum += abs(self.matrix[p, suit, rank])
                
        # Set target player to requested value first
        self.matrix[player, suit, rank] = absolute_value
        
        # Calculate remaining probability for others
        remaining_for_others = 1.0 - abs(absolute_value)
        
        # If there's no absolute probability to redistribute, distribute equally
        if abs_others_sum < 1e-6:
            # Distribute remaining probability equally among others
            per_player = remaining_for_others / 3.0
            
            for p in range(4):
                if p != player:
                    self.matrix[p, suit, rank] = per_player
            
            return True

        # Redistribute proportionally among other players, preserving signs
        for p in range(4):
            if p != player:
                # Get sign of current value
                sign = 1.0 if self.matrix[p, suit, rank] >= 0 else -1.0
                
                # Calculate proportion of total absolute sum
                abs_value = abs(self.matrix[p, suit, rank])
                proportion = abs_value / abs_others_sum
                
                # Calculate new absolute value
                new_abs_value = remaining_for_others * proportion
                
                # Apply sign to new value
                self.matrix[p, suit, rank] = sign * new_abs_value
        
        return True
    
    def extract(self, player, suit, rank, total_percentage_of_others):
        """
        Extract a percentage from other players and add it to the target player.
        Takes a percentage of the total probability of other players.
        
        Args:
            player: Player index (0-3)
            suit: Suit index (0-3)
            rank: Rank index (0-7)
            total_percentage_of_others: Percentage (0-1) of other players' total to extract
            
        Returns:
            success: Whether the extraction was successful
        """
        # Validate indices
        self._validate_indices(player, suit, rank)
        
        # Validate percentage
        if not (0 <= total_percentage_of_others <= 1):
            raise ValueError(f"Invalid percentage: {total_percentage_of_others}. Must be between 0 and 1")
        
        # Check if card has been played (-1.0)
        if any(self.matrix[p, suit, rank] == -1.0 for p in range(4)):
            return False  # Can't modify probabilities of played cards
        
        # Calculate total probability of other players
        others_sum = 0.0
        for p in range(4):
            if p != player:
                others_sum += self.matrix[p, suit, rank]
        
        # If others have no probability, nothing to extract
        if others_sum < 1e-6:
            return False
        
        # Calculate amount to extract
        extract_amount = others_sum * total_percentage_of_others
        
        # Calculate new absolute value for target player
        current_value = self.matrix[player, suit, rank]
        new_value = current_value + extract_amount
        
        # Use update to set the new value and redistribute remainder
        return self.update(player, suit, rank, new_value)
        
    def _validate_indices(self, player, suit, rank):
        if not (0 <= player < 4):
            raise ValueError(f"Invalid player index: {player}. Must be between 0 and 3")
        
        if not (0 <= suit < 4):
            raise ValueError(f"Invalid suit index: {suit}. Must be between 0 and 3")
        
        if not (0 <= rank < 8):
            raise ValueError(f"Invalid rank index: {rank}. Must be between 0 and 7")
        
    def to_tensor(self):
        return torch.tensor(self.matrix, dtype=torch.float32)
    
    def rotate(self, player_step=0, suit_step=0, rank_step=0):
        """Rotate the matrix by shifting indices."""
        matrix = np.zeros((4, 4, 8), dtype=np.float32)
        
        for player in range(4):
            for suit in range(4):
                for rank in range(8):
                    new_player = (player + player_step) % 4
                    new_suit = (suit + suit_step) % 4
                    new_rank = (rank + rank_step) % 8
                    matrix[new_player, new_suit, new_rank] = self.matrix[player, suit, rank]
        
        self.matrix = matrix
        return self
    
    def change_suits(self, transform_func):
        """Transform suits using the provided transformation function."""
        # Create a copy of the current matrix to read from
        original_matrix = np.copy(self.matrix)
        
        # Apply transformation
        for player in range(4):
            for suit in range(4):
                for rank in range(8):
                    new_suit = transform_func(suit)
                    self.matrix[player, new_suit, rank] = original_matrix[player, suit, rank]
        
        return self

    def copy(self):
        """Create a deep copy of this Probability object."""
        new = Probability()
        new.matrix = np.copy(self.matrix)
        return new
    
    def __getstate__(self):
        """Convert NumPy array to a regular list for better compatibility."""
        return {
            "matrix": self.matrix.tolist()
        }
    
    def __setstate__(self, state):
        """Convert the list back to a NumPy array with the correct dtype."""
        self.matrix = np.array(state["matrix"], dtype=np.float32)