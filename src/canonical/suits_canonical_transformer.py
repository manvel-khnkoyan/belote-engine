import numpy as np
from src.states.probability import Probability

class SuitsCanonicalTransformer:
    """
    Creates a canonical mapping for suits based on their global strength across all players.
    
    This class determines a standardized ordering of suits by:
    1. Calculating the global strength of each suit across all players based on probabilities
    2. Sorting suits by their global strength (strongest first)
    3. Creating a mapping from original suit indices to canonical indices
    """
    
    def __init__(self, probability: Probability):
        """
        Initialize the transformer with a probability matrix.
        
        Args:
            probability: A Probability object containing the current
                       probability distribution of cards.
        
        Raises:
            ValueError: If probability is None or has invalid matrix.
        """
        if probability is None or probability.matrix is None:
            raise ValueError("Invalid probability matrix")
        
        self.probability = probability
        self.matrix = probability.matrix
        self.num_players, self.num_suits, self.num_ranks = self.matrix.shape
        self._suit_map = None
        self._reverse_map = None
        self._calculate_mapping()
    
    def _calculate_mapping(self):
        """Calculate the suit mapping based on global strength."""
        # Calculate global suit strength across all players
        global_suit_strength = np.zeros(self.num_suits)
        
        for suit in range(self.num_suits):
            for player in range(self.num_players):
                # Sum positive probabilities for this player and suit across all ranks
                positive_probs = np.maximum(0, self.matrix[player, suit, :])
                global_suit_strength[suit] += np.sum(positive_probs)
        
        # Sort suits by global strength (strongest first)
        suits_with_strength = [(suit, global_suit_strength[suit]) for suit in range(self.num_suits)]
        suits_with_strength.sort(key=lambda x: -x[1])  # Sort by descending strength
        ordered_suits = [suit for suit, _ in suits_with_strength]
        
        # Create the mapping from original suit index to canonical suit index
        self._suit_map = {}
        for canonical_idx, original_idx in enumerate(ordered_suits):
            self._suit_map[original_idx] = canonical_idx
        
        # Create reverse mapping
        self._reverse_map = {v: k for k, v in self._suit_map.items()}
    
    def transform(self, suit_index):
        """
        Transform a suit index to its canonical form.
        
        Args:
            suit_index: Original suit index to transform.
            
        Returns:
            int: Canonical suit index.
            
        Raises:
            ValueError: If suit_index is invalid.
        """
        if not (0 <= suit_index < self.num_suits):
            raise ValueError(f"Invalid suit index: {suit_index}. Must be between 0 and {self.num_suits - 1}")
        
        return self._suit_map[suit_index]
    
    def reverse(self, canonical_index):
        """
        Reverse transform a canonical suit index back to original form.
        
        Args:
            canonical_index: Canonical suit index to reverse.
            
        Returns:
            int: Original suit index.
            
        Raises:
            ValueError: If canonical_index is invalid.
        """
        if not (0 <= canonical_index < self.num_suits):
            raise ValueError(f"Invalid canonical index: {canonical_index}. Must be between 0 and {self.num_suits - 1}")
        
        return self._reverse_map[canonical_index]
    
    def get_transform_function(self):
        """
        Get a lambda function for transforming suit indices.
        
        Returns:
            function: Lambda function that transforms suit indices.
        """
        return lambda x: self.transform(x)
    
    def get_reverse_function(self):
        """
        Get a lambda function for reverse transforming suit indices.
        
        Returns:
            function: Lambda function that reverse transforms suit indices.
        """
        return lambda x: self.reverse(x)
    
    def get_mapping(self):
        """
        Get the complete suit mapping dictionary.
        
        Returns:
            dict: Dictionary mapping original suit indices to canonical indices.
        """
        return self._suit_map.copy()
    
    def get_reverse_mapping(self):
        """
        Get the complete reverse suit mapping dictionary.
        
        Returns:
            dict: Dictionary mapping canonical suit indices to original indices.
        """
        return self._reverse_map.copy()
    
    def get_suit_strengths(self):
        """
        Get the calculated global strength for each suit.
        
        Returns:
            dict: Dictionary mapping original suit indices to their global strengths.
        """
        global_suit_strength = np.zeros(self.num_suits)
        
        for suit in range(self.num_suits):
            for player in range(self.num_players):
                positive_probs = np.maximum(0, self.matrix[player, suit, :])
                global_suit_strength[suit] += np.sum(positive_probs)
        
        return {suit: strength for suit, strength in enumerate(global_suit_strength)}
    
    def apply_to_probability(self, probability: Probability = None):
        """
        Apply the canonical transformation to a probability matrix.
        
        Args:
            probability: Probability object to transform. If None, uses the original.
            
        Returns:
            Probability: The transformed probability object (modifies in place).
        """
        if probability is None:
            probability = self.probability
        
        return probability.change_suits(self.get_transform_function())
    
    def reverse_probability(self, probability: Probability = None):
        """
        Reverse the canonical transformation on a probability matrix.
        
        Args:
            probability: Probability object to reverse transform. If None, uses the original.
            
        Returns:
            Probability: The reverse transformed probability object (modifies in place).
        """
        if probability is None:
            probability = self.probability
        
        return probability.change_suits(self.get_reverse_function())
    
    def __str__(self):
        """String representation of the transformer."""
        strengths = self.get_suit_strengths()
        mapping = self.get_mapping()
        
        result = "SuitsCanonicalTransformer:\n"
        result += "Suit Strengths (original -> strength):\n"
        for suit in range(self.num_suits):
            result += f"  Suit {suit}: {strengths[suit]:.4f}\n"
        
        result += "Canonical Mapping (original -> canonical):\n"
        for original, canonical in mapping.items():
            result += f"  {original} -> {canonical}\n"
        
        return result
    
    def __repr__(self):
        """Representation of the transformer."""
        return f"SuitsCanonicalTransformer(mapping={self.get_mapping()})"