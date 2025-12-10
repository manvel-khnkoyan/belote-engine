from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from .state import State
from .actions import Action

class Agent(ABC):
    
    @abstractmethod
    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Dict[str, Any] | None]:
        """
        Choose an action based on the current game state.
        
        Args:
            state (State): The current state of the game.
            actions (List[Action]): The list of possible actions.

        Returns:
            Tuple[Action, log]: A tuple containing the action chosen by the agent and a logs that can be used for training.
        """
        pass
