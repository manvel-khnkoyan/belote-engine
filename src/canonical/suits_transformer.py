import numpy as np
from src.states.probability import Probability

def suits_canonical_transformer(probability: Probability):
    """
    Creates a canonical mapping for suits based on their global strength across all players.
    
    This function determines a standardized ordering of suits by:
    1. Calculating the global strength of each suit across all players based on probabilities
    2. Sorting suits by their global strength (strongest first)
    3. Creating a mapping from original suit indices to canonical indices
    
    Args:
        probability: A Probability object containing the current
                           probability distribution of cards.
    
    Returns:
        dict: A dictionary mapping original suit indices to canonical suit indices.
    """
    # Input validation
    if probability is None or probability.matrix is None:
        raise ValueError("Invalid probability matrix")
    
    matrix = probability.matrix
    
    # Extract dimensions from the matrix
    num_players, num_suits, _ = matrix.shape
    
    # Calculate global suit strength across all players
    global_suit_strength = np.zeros(num_suits)
    for suit in range(num_suits):
        for player in range(num_players):
            # Sum positive probabilities for this player and suit across all ranks
            positive_probs = np.maximum(0, matrix[player, suit, :])
            global_suit_strength[suit] += np.sum(positive_probs)
    
    # Sort suits by global strength (strongest first)
    # Using a direct approach rather than argsort
    suits_with_strength = [(suit, global_suit_strength[suit]) for suit in range(num_suits)]
    suits_with_strength.sort(key=lambda x: -x[1])  # Sort by descending strength
    ordered_suits = [suit for suit, _ in suits_with_strength]
    
    # Create the mapping from original suit index to canonical suit index
    suit_map = {}
    for canonical_idx, original_idx in enumerate(ordered_suits):
        suit_map[original_idx] = canonical_idx
    
    # Transform lambda function to apply the mapping
    transform = lambda x: suit_map[x]

    # Reverse lambda function to revert the mapping
    reverse = lambda x: {v: k for k, v in suit_map.items()}.get(x, x)

    return transform, reverse