from abc import ABC, abstractmethod
from typing import List
from .state import State
from .actions import Action

class Agent(ABC):
    
    @abstractmethod
    def choose_action(self, state: State, actions: List[Action]) -> Action:
        """
        Choose an action based on the current game state.
        
        Args:
            state (State): The current state of the game.
            actions (List[Action]): The list of possible actions.

        Returns:
            Action: The action chosen by the agent.
        """
        pass
