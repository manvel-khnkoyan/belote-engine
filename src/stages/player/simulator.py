from src.stages.player.helper_agents.human import Human

def default_action_selector(env, agent):
    """Default action selector - just calls the agent's choose_action method"""
    return agent.choose_action(env)

def play_trick(env, agents, display, action_selector):
    """Play a single trick (4 cards, one from each player)"""
    while True:
        current_player = env.next_player
        agent = agents[current_player]

        # Display game state for human players
        show_game_state(env, current_player, display)

        # Get and execute the action
        action = action_selector(env, agent)
        _, trick_ended, round_ended = env.step(action)

        # Notify all agents about the action
        for i in range(4):
            agents[i].observe((current_player + i) % 4, action)
        
        if trick_ended:
            return round_ended

def simulate(env, agents, action_selector=None, display=False, on_trick_end=None, on_round_end=None):
    """Simulate a complete round of Belote."""
    # Use default action selector if none provided
    if action_selector is None:
        action_selector = default_action_selector
    
    # Initialize all agents
    for i, agent in enumerate(agents):
        agent.init(env, env_index=i)
    
    # Play tricks until round ends
    while True:
        env.reset_trick()
        
        round_ended = play_trick(env, agents, display, action_selector)
        
        if on_trick_end:
            on_trick_end()

        show_trick_result(env, display)

        if round_ended:
            break

    # Round ended
    if on_round_end:
        on_round_end()


    show_round_result(env, display)

def show_game_state(env, player, display):
    if display == True or (display == player and type(display) is int):
        """Display the current game state for a player"""
        env.display_line()
        env.display_state()
        env.display_hands(player=player)
        env.display_table(end=" ")
        env.display_available_cards(player=player, end=" " if type(display) is int else None)

def show_trick_result(env, display):
    """Display the completed trick"""
    if display != False:
        env.display_table()

def show_round_result(env, display):
    """Display the final round summary"""
    if display != False:
        env.display_line()
        env.display_summary()