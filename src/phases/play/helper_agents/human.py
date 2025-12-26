
from src.phases.play.core.state import State
from typing import List, Tuple, Dict, Any
from src.phases.play.core.agent import Agent
from src.phases.play.core.actions import Action, ActionPlayCard

class HumanAgent(Agent):
    """Human player agent that prompts for input"""
    
    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Dict[str, Any] | None]:
        """Prompt human player to choose an action"""
        # Filter valid play card actions
        valid_play_actions = [action for action in actions if isinstance(action, ActionPlayCard)]
        
        if not valid_play_actions:
            # If no play card actions, return the first available action if any
            return actions[0] if actions else None
        
        valid_cards = [action.card for action in valid_play_actions]
        player_hand = state.cards
        
        # Display available cards
        self._display_options(player_hand, valid_cards)
        
        # Get user input
        return self._get_user_choice(player_hand, valid_cards, valid_play_actions), None
    
    def _display_options(self, player_hand: List, valid_cards: List):
        """Display player's hand with valid card indices"""
        card_indices = []
        for i, card in enumerate(player_hand):
            if card in valid_cards:
                card_indices.append(str(i + 1))
            else:
                card_indices.append('.')
        
        print(f"Hands:[{' '.join(str(card) for card in player_hand)}]")
        print(f"Input:  {'   '.join(card_indices)}", end=" : ")
    
    def _get_user_choice(self, player_hand: List, valid_cards: List, valid_actions: List[ActionPlayCard]) -> Action:
        """Get and validate user's card choice"""
        # Create mapping of indices to cards
        card_map = {i + 1: card for i, card in enumerate(player_hand)}
        
        while True:
            try:
                choice_input = input()
                if not choice_input:
                    continue
                choice = int(choice_input)
                if choice in card_map and card_map[choice] in valid_cards:
                    # Find the action corresponding to this card
                    selected_card = card_map[choice]
                    for action in valid_actions:
                        if action.card == selected_card:
                            return action
                else:
                    print("Invalid choice. Enter a valid number: ", end="")
            except ValueError:
                print("Please enter a number: ", end="")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                raise

