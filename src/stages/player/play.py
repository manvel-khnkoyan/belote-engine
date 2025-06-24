import os
import argparse
import time
import numpy as np
from src.stages.player.network import CNNBeloteNetwork
from src.stages.player.agent import PPOBeloteAgent
from src.stages.player.human import Human
from src.stages.player.history import History
from src.deck import Deck
from src.states.trump import Trump
from src.stages.player.env import BeloteEnv
from src.stages.player.history import History
from src.stages.player.simulation import play, test


def parse_args():
    """"""
    root = os.environ.get("PROJECT_ROOT")
    
    history_dir = os.path.join(root, 'histories')
    os.makedirs(history_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Play Belote against trained AI agents")
    parser.add_argument("--model-path", type=str, default=os.path.join(root, 'models', 'belote_agent_final.pt'), help="Model path for the agent")
    parser.add_argument("--mode", type=str, default='play', choices=['play', 'observe', 'record', 'replay', 'test'], help="Type of run: play, observe, record, replay, test")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the game n times")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to play")
    parser.add_argument("--history-dir", type=str, default=history_dir, help="Histories root directory")
    parser.add_argument("--history-file", type=str, default="history-20250515-160742", help="History file name")
    
    return parser.parse_args()

def create_env():
    """"""
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

def load_ai_agent(model_path):
    """"""
    network = CNNBeloteNetwork()
    agent = PPOBeloteAgent(network=network)
    agent.load(model_path)
    
    return agent

def load_human_agent(model_path):
    """"""
    network = CNNBeloteNetwork()
    agent = Human(network=network)
    agent.load(model_path)
    
    return agent

def get_history_files(history_dir):
    """"""
    items = os.listdir(history_dir)
    files = [item for item in items if item.startswith('history-') and os.path.isfile(os.path.join(history_dir, item))]

    return files

def initialize_agents(env, agents):
    """Initialize all agents with their environment and position index"""
    for i, agent in enumerate(agents):
        agent.init(env, env_index=i)

def fn_record(args):
    """"""
    env = create_env()

    agents = []
    for i in range(0, 4):
        agent = load_human_agent(args.model_path)
        agents.append(agent)

    # Initialize all agents with their environment index
    initialize_agents(env, agents)

    history = History(env)

    _, _, history = play(env, agents, history, display=True)
    
    # Ensure history directory exists
    if not os.path.exists(args.history_dir):
        os.makedirs(args.history_dir)
        print(f"Created history directory: {args.history_dir}")
    
    history_dir = args.history_dir
    history_path =  os.path.join(history_dir, f"history-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}")

    history.save(history_path)
    print(f"Saved history to {history_path}")

def fn_observe(args):
    """"""
    total = 0
    total_wins = 0

    agents = []
    for _ in range(0, 4):
        agent = load_ai_agent(args.model_path)
        agents.append(agent)

    while total < args.episodes:
        env = create_env()
        
        # Initialize all agents with their environment index
        initialize_agents(env, agents)
        
        history = History(env)
        
        gain, lose, _ = play(env, agents, history, display=False)  # Pass history object

        total += 1
        total_wins += (gain > lose) * 1
    
    print(f"Total played {total} | Wins: {total_wins} / Loses {total - total_wins}")

def fn_test(args):
    """"""
    # Create agents
    agents = []
    for i in range(0, 4):
        agent = load_ai_agent(args.model_path)
        agents.append(agent)

    total_moves = 0
    right_moves = 0

    # Loop through history files
    history_files = get_history_files(args.history_dir)
    if not history_files:
        print("No history files found. Cannot perform testing.")
        return
        
    print(f"Testing on {len(history_files)} history files...")
    
    for history_file in history_files:
        history = History()
        history_path = os.path.join(args.history_dir, history_file)
        history.load(history_path)
        env = history.load_env()

        # Initialize all agents with their environment index
        initialize_agents(env, agents)

        # Correct the call to test()
        _r, _t = test(env, agents, history)

        right_moves += _r
        total_moves += _t
        
        print(f"File: {history_file}, Correct moves: {_r}/{_t}")

    if total_moves > 0:
        accuracy = right_moves / total_moves
        accuracy_percent = round(accuracy * 100, 2)
        print(f"Overall accuracy: {accuracy_percent}% , {right_moves}/{total_moves}")

        return accuracy
    else:
        print("No moves were tested.")

def fn_replay(args):
    """"""
    history = History()
    history_path = os.path.join(args.history_dir, args.history_file)
    if not os.path.exists(history_path):
        print(f"Error: History file {history_path} not found")
        return
    
    history.load(history_path)
    env = history.load_env()

    print(f"Successfully loaded history from {history_path}")

    agents = []
    for i in range(0, 4):
        agent = load_ai_agent(args.model_path)
        agents.append(agent)

    # Initialize all agents with their environment index
    initialize_agents(env, agents)

    print(f"Replaying game from history file: {args.history_file}")
    test(env, agents, history, display=True)

def fn_play(args):
    """"""
    env = create_env()

    agents = []
    for i in range(0, 4):
        agent = load_human_agent(args.model_path) if i == 0 else load_ai_agent(args.model_path)
        agents.append(agent)

    # Initialize all agents with their environment index
    initialize_agents(env, agents)

    play(env, agents, display=True)

if __name__ == "__main__":
    """"""
    args = parse_args()
    
    # Validate history file argument for replay mode
    if args.mode == 'replay' and args.history_file is None:
        print("Error: --history-file argument is required for replay mode")
        exit(1)

    # Human playing against agents
    if args.mode == 'play': # Default
        fn_play(args)
    
    # Agents play with each other
    if args.mode == 'observe':
        fn_observe(args)

    # Humans play with each other to record new history
    if args.mode == 'record':
        fn_record(args)
    
    # Agent play reply the history / history=FILENAME
    if args.mode == 'replay':
        fn_replay(args)    
        
    # Testing all the recorded history
    if args.mode == 'test':
        repeat_count = args.repeat
        total_accuracy = 0

        for i in range(0, repeat_count):
            total_accuracy += fn_test(args)

        if repeat_count > 1:
            accuracy =  total_accuracy / repeat_count
            accuracy_percent = round(accuracy * 100, 2)
            print(f"Total {repeat_count} repetition accuracy: {accuracy_percent}%")