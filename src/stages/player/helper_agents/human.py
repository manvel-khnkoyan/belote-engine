
from src.card import Card
from src.stages.player.ppo.belote_agent import PPOBeloteAgent
from src.stages.player.actions import ActionCardPlay

class Human(PPOBeloteAgent):
    def choose_action(self, env):
        recomended_action = super().choose_action(env)

        recomended_action_index = None
        for i, card in enumerate(env.deck[self.env_index]):
            if card == recomended_action.card:
                recomended_action_index = i

        if isinstance(recomended_action, ActionCardPlay):
            print(f"Recommended:{recomended_action.card} ({recomended_action_index + 1})", end=" ")

        # Get the selected card from the human player
        action = self._get_selected_actions(env)
        
        return action

    def _get_selected_actions(self, env):
        valid_cards = env.valid_cards()
        
        # Create a mapping of card indices to cards
        card_map = {}
        for i, card in enumerate(env.deck[self.env_index]):
            card_map[i+1] = card  # Map 1-based indices to cards
        
        # Get human input
        while True:
            try:
                choice = int(input())
                if choice in card_map and card_map[choice] in valid_cards:
                    return ActionCardPlay(card_map[choice])
                else:
                    print("Invalid choice. Enter a number from your hand for a valid card: ", end="")
            except ValueError:
                print("Please enter a valid number: ", end="")

