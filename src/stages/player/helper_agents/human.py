
from src.card import Card
from src.stages.player.ppo.belote_agent import PPOBeloteAgent
from src.stages.player.actions import ActionCardPlay

class Human(PPOBeloteAgent):
    def choose_action(self, env):
        valid_cards = env.valid_cards()
        recomended_action = super().choose_action(env)

        recomended_action_index = None
        for i, card in enumerate(env.deck[self.env_index]):
            if card == recomended_action.card:
                recomended_action_index = i

        player_cards = env.deck[self.env_index]
        available_cards_indices = self._get_available_cards(env, valid_cards)
        card_indexes = [str(i + 1) if i + 1 in available_cards_indices else '.' for i in range(len(player_cards))]

        if isinstance(recomended_action, ActionCardPlay):
            print(f"Input:  {'   '.join(card_indexes)} | {recomended_action.card} ({recomended_action_index + 1}) :", end=" ")

        # Get the selected card from the human player
        action = self._get_selected_actions(env, valid_cards)
        print()
        
        return action

    def _get_selected_actions(self, env, valid_cards):
        
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

    def _get_available_cards(self, env, valid_cards):
        return [i + 1 for i, card in enumerate(env.deck[env.next_player]) if card in valid_cards]

