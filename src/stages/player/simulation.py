from src.stages.player.human import Human

def play(env, agents, history=None, display=False):
    # Main game loop
    round_ended = False

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

            if display and isinstance(agent, Human):
                env.display_table(end=" ")
                env.display_available_cards(player=current_player, end=" ")
            
            # Get the valid cards for the current player
            action = agent.choose_action(env)

            if history is not None:
                history.write(current_player, action)

            # Check if the action is valid
            _, trick_ended, round_ended = env.step(action)

            # Observe the action for all players
            for i in range(4):
                agents[i].observe((current_player + i) % 4, action)
        
        if display:
            env.display_table()
    
    if display:
        env.display_summary()

    return env.round_scores[0], env.round_scores[1], history

def test(env, agents, history, display=False):
    total_moves=0
    total_right=0

    # Main game loop
    round_ended = False

    # Generate Env
    env = history.create_env()

    for i in range(4):
        agents[i].init(history, env_index=i)
    
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
                env.display_table(end="")
                env.display_available_cards(player=current_player)
            
            # Get the valid cards for the current player
            action = history.read()
            agent_action = agent.choose_action(env)

            total_moves += 1
            if agent_action == action:
                total_right += 1

            # Check if the action is valid
            _, trick_ended, round_ended = env.step(action)

            # Observe the action for all players
            for i in range(4):
                agents[i].observe((current_player + i) % 4, action)
        
        if display:
            env.display_table()
    
    if display:
        env.display_summary()

    return env.round_scores[0], env.round_scores[1], history
