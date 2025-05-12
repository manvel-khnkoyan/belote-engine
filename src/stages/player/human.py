from src.types import ACTION_TYPE_CARD

class Human:
    def __init__(self):
        self.env = None

    def init(self, env, env_index):
        self.env_index = env_index
        return None

    def observe(self, player_index, action):
        return None

    def choose_action(self, env):        
        # Get the selected card from the human player
        card = self._get_selected_card(env)
        
        return {
            'type': ACTION_TYPE_CARD,
            'move': card
        }

    def _get_selected_card(self, env):
        valid_cards = env.valid_cards()
        
        if not valid_cards:
            print("You have no valid cards to play!")
            return None
        
        # Create a mapping of card indices to cards
        card_map = {}
        for i, card in enumerate(env.deck[self.env_index]):
            card_map[i+1] = card  # Map 1-based indices to cards
        
        # Get human input
        while True:
            try:
                choice = int(input())
                if choice in card_map and card_map[choice] in valid_cards:
                    return card_map[choice]
                else:
                    print("Invalid choice. Enter a number from your hand for a valid card: ", end="")
            except ValueError:
                print("Please enter a valid number: ", end="")

