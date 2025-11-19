
from typing import List
from src.models import Trump, Table, Probability, Deck
from src.tricks.actions import Action, ActionPlayCard
from src.tricks.agents.agent import Agent


class HumanAgent(Agent):
    """Human player agent that prompts for input"""
    
    def choose_action(self, player: int, trump: Trump, table: Table, probability: Probability, valid_actions: List[Action]) -> Action:
        """Prompt human player to choose an action"""
        # Filter valid play card actions
        valid_play_actions = [action for action in valid_actions if isinstance(action, ActionPlayCard)]
        
        if not valid_play_actions:
            return valid_actions[0] if valid_actions else None
        
        # Get player's hand from probability (cards with probability 1.0 for this player)
        player_hand = self._get_player_hand(player, probability)
        valid_cards = [action.card for action in valid_play_actions]
        
        # Display available cards
        self._display_options(player_hand, valid_cards)
        
        # Get user input
        return self._get_user_choice(player_hand, valid_cards)
    
    def _get_player_hand(self, player: int, probability: Probability) -> List:
        """Extract player's hand from probability matrix"""
        hand = []
        all_cards = Deck.new_cards()
        for card in all_cards:
            if probability.matrix[player, card.suit, card.rank] == 1.0:
                hand.append(card)
        return hand
    
    def _display_options(self, player_hand: List, valid_cards: List):
        """Display player's hand with valid card indices"""
        card_indices = []
        for i, card in enumerate(player_hand):
            if card in valid_cards:
                card_indices.append(str(i + 1))
            else:
                card_indices.append('.')
        
        print(f"Hands: {' '.join(str(card) for card in player_hand)}")
        print(f"Input: {'   '.join(card_indices)}", end=" : ")
    
    def _get_user_choice(self, player_hand: List, valid_cards: List) -> Action:
        """Get and validate user's card choice"""
        # Create mapping of indices to cards
        card_map = {i + 1: card for i, card in enumerate(player_hand)}
        
        while True:
            try:
                choice = int(input())
                if choice in card_map and card_map[choice] in valid_cards:
                    return ActionPlayCard(card_map[choice])
                else:
                    print("Invalid choice. Enter a valid number: ", end="")
            except ValueError:
                print("Please enter a number: ", end="")
            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl+C or EOF gracefully
                print("\nExiting...")
                raise

