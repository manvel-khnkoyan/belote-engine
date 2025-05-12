import numpy as np
from src.types import ACTION_TYPE_CARD
from src.deck import Deck
from src.states.trump import Trump
from src.stages.player.env import BeloteEnv
from src.stages.player.history import History

def create_env():
    trump = Trump()
    trump.set_random_trump()

    deck = Deck()
    deck.reset()
    deck.deal_cards(8)
    deck.reorder_hands(trump)

    # Choose random starting player
    rng = np.random.default_rng(None)
    next_player = int(rng.integers(0, 4))
    
    # Create the environment
    return BeloteEnv(trump, deck, next_player)

def simulate(env, agents, record=False, verbose=None):
    # Main game loop
    round_ended = False
    history = History(env) if record else None

    for i in range(4):
        agents[i].init(env, env_index=i)
    
    while not round_ended:
        # Reset the trick
        env.reset_trick()

        if verbose is not None:
            env.display_state()

        trick_ended = False
        while not trick_ended:
            current_player = env.next_player

            if verbose is not None and current_player in verbose:
                env.display_table()
            
            # Get the valid cards for the current player
            action = agents[current_player].choose_action(env)

            if history is not None:
                history.record(current_player, action)

            # Check if the action is valid
            _, trick_ended, round_ended = env.step(action)

            # Observe the action for all players
            for i in range(4):
                agents[i].observe((current_player + i) % 4, action)
        
        if verbose is not None:
            env.display_table()
    
    if verbose is not None:
        env.display_summary()

    return env.round_scores[0], env.round_scores[1], history
