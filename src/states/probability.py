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
        self.matrix = np.zeros((4, 4, 8), dtype=np.float32)

    def reset(self, deck, my_player_cards):
        """Reset the probability matrix to all zeros."""
        self.matrix = np.zeros((4, 4, 8), dtype=np.float32)

        for player in range(4):
            for (suit, rank) in deck:
                if (suit, rank) in my_player_cards[0]:
                    self.matrix[player, suit, rank] = 1 if player == 0 else 0
                else:
                    self.matrix[player, suit, rank] = 1/3
        
    
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
    def update(self, player, suit, rank, absolute_value):
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
            self.matrix[player, suit, rank] = round(absolute_value)
            return True
        
        # Get current values
        current_value = self.matrix[player, suit, rank]
        
        # Calculate difference that needs to be redistributed
        difference = current_value - absolute_value
        
        # If no difference, no need to update
        if abs(difference) < 1e-6:
            return True
        
        # Calculate sum of absolute probabilities of other players
        abs_others_sum = 0
        for p in range(4):
            if p != player:
                abs_others_sum += abs(self.matrix[p, suit, rank])
                
        # If there's no absolute probability to redistribute, handle specially
        if abs_others_sum < 1e-6 and difference != 0:
            # Set target player to requested value
            self.matrix[player, suit, rank] = absolute_value
            
            # Distribute remaining probability equally among others
            remaining = 1.0 - abs(absolute_value)
            per_player = remaining / 3
            
            for p in range(4):
                if p != player:
                    self.matrix[p, suit, rank] = per_player
            
            return True
        
        # Update the target player's probability
        self.matrix[player, suit, rank] = absolute_value
        
        # Redistribute the difference among other players proportionally,
        # preserving their signs
        if abs_others_sum > 0:
            for p in range(4):
                if p != player:
                    # Get sign of current value
                    sign = 1 if self.matrix[p, suit, rank] >= 0 else -1
                    
                    # Calculate proportion of total absolute sum
                    abs_value = abs(self.matrix[p, suit, rank])
                    proportion = abs_value / abs_others_sum
                    
                    # Calculate new absolute value
                    new_abs_value = abs_value - abs(difference) * proportion
                    
                    # Apply sign to new value
                    self.matrix[p, suit, rank] = sign * new_abs_value
        
        # Ensure sum of absolute values equals 1.0
        abs_total = sum(abs(self.matrix[p, suit, rank]) for p in range(4))
        
        if abs(abs_total - 1.0) > 1e-6 and abs_total > 0:
            # Normalize all values to make absolute sum equal to 1
            for p in range(4):
                self.matrix[p, suit, rank] = self.matrix[p, suit, rank] / abs_total
        
        return True
    
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
    def extract(self, player, suit, rank, total_percentage_of_others):
        # Validate indices
        self._validate_indices(player, suit, rank)
        
        # Validate percentage
        if not (0 <= total_percentage_of_others <= 1):
            raise ValueError(f"Invalid percentage: {total_percentage_of_others}. Must be between 0 and 1")
        
        # Check if card has been played (-1.0)
        if any(self.matrix[p, suit, rank] == -1.0 for p in range(4)):
            return False  # Can't modify probabilities of played cards
        
        # Calculate total probability of other players
        others_sum = 0
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
        return torch.tensor(self.matrix, dtype=torch.float)
    
    def transform_matrix(self, player_step = 0, suit_step = 0, rank_step = 0):
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
    
    def change_suits(self, transform):
        for player in range(4):
            for suit in range(4):
                for rank in range(8):
                    self.matrix[player, transform(suit), rank] = self.matrix[player, suit, rank]
        
        return self

    def copy(self):
        new = Probability()
        new.matrix = np.copy(self.matrix)
        
        return new