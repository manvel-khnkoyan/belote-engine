"""
Interactive play script for Belote with simplified 4-line interface.
"""

import os
import argparse
import numpy as np
from src.stages.player.network import CNNBeloteNetwork
from src.stages.player.agent import PPOBeloteAgent
from src.stages.player.human import Human
from src.stages.player.history import History
from src.deck import Deck
from src.states.trump import Trump
from src.stages.player.env import BeloteEnv
from src.stages.player.history import History
from src.stages.player.simulation import play

def parse_args():
    parser = argparse.ArgumentParser(description="Play Belote against trained AI agents")
    parser.add_argument("--model-path", type=str, default="./models/belote_agent_final.pt", help="Model path for the agent")
    parser.add_argument("--mode", type=str, default='play', choices=['play', 'observe', 'record', 'test'], help="Type of run: play, observe, record")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to play")
    return parser.parse_args()

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

def load_ai_agent(model_path):
    network = CNNBeloteNetwork()
    agent = PPOBeloteAgent(network=network)
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    
    return agent

def load_human_agent(model_path):
    network = CNNBeloteNetwork()
    agent = Human(network=network)
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    
    return agent


def play_to_record(args):
    env = create_env()

    agents = []
    for i in range(0, 4):
        agent = load_human_agent(args.model_path)
        agent.reset(env, player=i)
        agents.append(agent)


    history = History(env)
    gain, lose, history = play(env, agents, history, display=True)
    history.save("path-to-save")

    print(f"You {"won :)" if gain >= lose else "lose :()"} | points: {gain} / {lose}")  


def play_to_observe(args):
    n = 0
    total_wins = 0
    agents = []
    for _ in range(0, 4):
        agent = load_ai_agent(args.model_path)
        agents.append(agent)

    while n < args.episodes:
        env = create_env()

        gain, lose, _ = play(env, agents, display=False)

        n = n + 1
        total_wins += (gain > lose) * 1
    
    print(f"Total played {n} | Wins: {total_wins} / {n - total_wins}")  


def play_against_agents(args):
    env = create_env()
    history = History(env)

    # Create a separate agent for each AI player
    agents = []
    for i in range(0, 4):
        agent = load_human_agent(args.model_path) if i == 0 else load_ai_agent(args.model_path)
        agents.append(agent)

    gain, lose, _ = play(env, agents, history, display=True)

    print(f"You {"win :)" if gain >= lose else "lose :("} | points: {gain} / {lose}")  

    history.save('./history_001')


if __name__ == "__main__":
    args = parse_args()

    # Different play modes
    match args.mode:
        case 'play':
            play_against_agents(args)
        case 'observe':
            play_to_observe(args)
        case 'record':
            play_to_record(args)
