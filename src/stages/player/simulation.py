from src.stages.player.helper_agents.human import Human
from src.stages.player.env import BeloteEnv
from src.stages.player.history import History

def play(env, agents, recorder:History=None, display=False):
    # Main game loop
    round_ended = False

    for i in range(4):
        agents[i].init(env, env_index=i)
    
    while not round_ended:
        # Reset the trick
        env.reset_trick()

        trick_ended = False
        while not trick_ended:
            current_player = env.next_player
            agent = agents[current_player]

            if display and isinstance(agent, Human):
                env.display_state()
                env.display_hands(player=current_player)

                env.display_table(end=" ")
                env.display_available_cards(player=current_player, end=" ")
            
            # Get the valid cards for the current player
            action = agent.choose_action(env)

            if recorder is not None:
                recorder.record_action(current_player, action)

            # Check if the action is valid
            _, trick_ended, round_ended = env.step(action)

            # Observe the action for all players
            for i in range(4):
                agents[i].observe((current_player + i) % 4, action)
        
        if display:
            env.display_table()
    
    if display:
        env.display_summary()

    return env.round_scores[0], env.round_scores[1]

def test(env, agents, history:History, display=False, ):
    total_moves=0
    right_moves=0

    # Main game loop
    round_ended = False
    history.reset()

    for i in range(4):
        agents[i].init(env, env_index=i)
    
    while not round_ended:
        # Reset the trick
        env.reset_trick()

        if display:
            env.display_state()

        trick_ended = False
        while not trick_ended:
            current_player = env.next_player
            agent = agents[current_player]

            if display:
                env.display_hands(player=current_player)
                env.display_table(end="")
                env.display_available_cards(player=current_player, end="")
            
            # Get the valid cards for the current player
            action, _ = history.get_next_action()
            agent_action = agent.choose_action(env)

            if display:
                print(action.card ,end="")
                input()

            valid_cards = env.valid_cards()
            # Count only whene there are multiple options
            if len(valid_cards) > 1:
                total_moves += 1
                if agent_action == action:
                    right_moves += 1

            # Check if the action is valid
            _, trick_ended, round_ended = env.step(action)

            # Observe the action for all players
            for i in range(4):
                agents[i].observe((current_player + i) % 4, action)
        
        if display:
            env.display_table()
    
    if display:
        env.display_summary()

    return right_moves, total_moves
