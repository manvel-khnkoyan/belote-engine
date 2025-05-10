"""
Interactive play script for Belote with simplified 4-line interface.
"""

import os
import argparse
import numpy as np
from src.stages.player.env import BeloteEnv
from src.stages.player.network import CNNBeloteNetwork
from src.stages.player.agent import PPOBeloteAgent
from src.states.trump import Trump
from src.card import SUIT_SYMBOLS
from src.deck import Deck

def parse_args():
    parser = argparse.ArgumentParser(description="Play Belote against trained AI agents")
    parser.add_argument("--model-path", type=str, default="./models/belote_agent_final.pt", 
                        help="Path to the trained model checkpoint (will be used for all agents if agent-specific paths not provided)")
    parser.add_argument("--agent1-model", type=str, default=None, 
                        help="Path to agent 1's model (if different)")
    parser.add_argument("--agent2-model", type=str, default=None, 
                        help="Path to agent 2's model (if different)")
    parser.add_argument("--agent3-model", type=str, default=None, 
                        help="Path to agent 3's model (if different)")
    parser.add_argument("--human-player", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Position of the human player (0-3)")
    parser.add_argument("--observe", action="store_true", 
                        help="Observe AI vs AI game (no human player)")
    parser.add_argument("--test-case", type=int, default=0, choices=[0, 1],
                        help="Special case when human plays alone to record the game for testing")
    return parser.parse_args()

def setup_game():
    """Set up a new game environment."""
    trump = Trump()
    trump.set_random_trump()
    
    deck = Deck()
    deck.reset()
    deck.deal_cards(8)
    deck.reorder_hands(trump)
    
    # Create environment with a random starting player (0-3)
    rng = np.random.default_rng(None)
    next_player = int(rng.integers(0, 4))  # Randomly choose who starts

    env = BeloteEnv(trump, deck, next_player=next_player)
    
    return env

def display_state(env):
    # Find the trump suit
    if np.any(env.trump.values == 1):
        trump_suit_index = np.argmax(env.trump.values)
        trump_suit = SUIT_SYMBOLS[trump_suit_index]
    else:
        trump_suit = "No"
        
    # Get player's cards
    player_cards = env.deck[0]
    game_round = 9 - len(player_cards)  # Belote has 8 rounds total (8 cards per player)

    print(f"Round: {game_round} | Starts: {env.next_player} | Trump: {trump_suit}")
    print(f"Hands: {' '.join([f"{str(player_cards[i])}" for i in range(len(player_cards))])}")
    print(f"Index: {''.join([f" {i + 1}  " for i in range(len(player_cards))])}")

def display_table(env, recommended_card=None, your_move=False):
    table_cards = []
    for card in env.table.cards:
        if card is not None:
            # Simply show all cards without indicating who played them
            table_cards.append(str(card))
    
    # Fill with placeholders if needed
    while len(table_cards) < 4:
        table_cards.append(".")
    
    table_str = " ".join(table_cards)
    
    available_cards = env.valid_cards()
    available_cards_indices = [str(i + 1) for i, card in enumerate(env.deck[0]) if card in available_cards]

    # Convert recommended_card to index if provided
    recommended_idx = None
    if recommended_card:
        for i, card in enumerate(env.deck[0]):
            if card == recommended_card:
                recommended_idx = i + 1
                break

    if your_move:
        rec_str = f"Recommended({recommended_idx})" if recommended_idx else ""
        print(f"Table: {table_str} | [{','.join(available_cards_indices)}] {rec_str} Your-Move: ", end="")
    else:
        print(f"Table: {table_str}")

def display_summary(env):
    print(f"Total: Last ({env.trick_scores[0]}, {env.trick_scores[1]}) Total ({env.round_scores[0]}, {env.round_scores[1]})")

def get_selected_card(env):
    """Get action from the human player."""
    valid_cards = env.valid_cards()
    
    if not valid_cards:
        print("You have no valid cards to play!")
        return None
    
    # Create a mapping of card indices to cards
    card_map = {}
    for i, card in enumerate(env.deck[0]):
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

def create_agent(model_path):
    """Load a model for an agent."""
    network = CNNBeloteNetwork()
    agent = PPOBeloteAgent(network=network)
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    
    return agent

def play_game(args):
    # Parse command line arguments
    human_player = args.human_player
    observe_mode = args.observe

    # Setup a new game environment
    env = setup_game()
    
    # Create a separate agent for each AI player
    ai_agents = []
    for i in range(4):
        agent = create_agent(args.model_path)
        agent.reset_memory(env, player=i)
        ai_agents.append(agent)

    # Main game loop
    round_ended = False
    
    while not round_ended:
        # Check if we need to start a new trick
        
        trick_ended = False
        while not trick_ended:
            current_player = env.next_player

            if env.table.is_empty():
                display_state(env)
            
            # Human player action
            if current_player == human_player and not observe_mode:
                # Get recommendation from the AI for the human player
                recommended_card = ai_agents[current_player].choose_action(env)

                # Display the game state with recommendation
                display_table(env, recommended_card, your_move=True)
                
                # Get human player's action
                card = get_selected_card(env)
            else:
                # Get AI's move
                card = ai_agents[current_player].choose_action(env)

            # Take the action
            _, trick_ended, round_ended = env.step(card)

            # Update all AI agents' memories with the move
            for agent in ai_agents:
                agent.update_memory(current_player, card)

            # If the trick has ended
            if trick_ended:
                display_table(env)
                display_summary(env)
                print("\n" + '-' * 50 + "\n")
                env.reset_trick()
                    
        env.reset_trick()
    
    print(f"You {'won :)' if env.round_scores[human_player] > env.round_scores[1-human_player] else 'lost :('}")


if __name__ == "__main__":
    args = parse_args()
    play_game(args)