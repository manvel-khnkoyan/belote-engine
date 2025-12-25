from typing import Any, Dict, List, Tuple
from src.phases.play.core.state import State
from src.phases.play.core.agent import Agent
from src.phases.play.core.actions import Action, ActionPlayCard
from src.utility.cards import Cards


class AggressivePlayerAgent(Agent):
    """Aggressive player agent that tries to win the trick"""
    
    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Dict[str, Any] | None]:
        """Choose the card that wins the trick, or highest value card"""
        # Filter for play card actions only
        play_actions = [action for action in actions if isinstance(action, ActionPlayCard)]
        
        if not play_actions:
            # If no play card actions available, return first action
            return actions[0] if actions else None, None
        
        # If leading (table empty), play highest value card
        if not state.table:
            best_action = max(play_actions, key=lambda a: a.card.value(state.trump))
            return best_action, None

        # Find current winner on table
        current_winner, _ = Cards.winner(state.table, state.trump)
        
        # Find cards that beat the current winner
        winning_actions = [a for a in play_actions if a.card.beats(state.trump, current_winner)]
        
        if winning_actions:
            # If we can win, play the highest value winning card (Aggressive)
            best_action = max(winning_actions, key=lambda a: a.card.value(state.trump))
        else:
            # If we cannot win, play the lowest value card (discard)
            best_action = min(play_actions, key=lambda a: a.card.value(state.trump))
        
        return best_action, None
