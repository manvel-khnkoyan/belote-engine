from abc import ABC, abstractmethod
from typing import List
from src.models import Trump, Table, Probability
from src.tricks.actions import Action


class Agent(ABC):
    """Base interface for all Belote game agents"""
    
    @abstractmethod
    def choose_action(self, player: int, trump: Trump, table: Table, probability: Probability, valid_actions: List[Action]) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            player: Current player index (0-3)
            trump: Current trump configuration
            table: Current table state with played cards
            probability: Probability matrix of card locations
            valid_actions: List of valid actions the player can take
            
        Returns:
            Action to take
        """
        pass
