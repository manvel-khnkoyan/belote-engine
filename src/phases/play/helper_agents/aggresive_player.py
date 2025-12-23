from typing import Any, Dict, List, Tuple
from src.phases.play.core.state import State
from src.phases.play.core.agent import Agent
from src.phases.play.core.actions import Action, ActionPlayCard


class AggressivePlayerAgent(Agent):
    """Aggressive player agent that always plays the highest card available"""
    
    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Dict[str, Any] | None]:
        """Choose the highest ranked card from valid actions"""
        # Filter for play card actions only
        play_actions = [action for action in actions if isinstance(action, ActionPlayCard)]
        
        if not play_actions:
            # If no play card actions available, return first action
            return actions[0] if actions else None
        
        # Find the card with the highest rank value
        highest_card_action = max(play_actions, key=lambda action: (action.card.rank, action.card.suit))
        
        return highest_card_action, None
