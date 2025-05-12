"""
Interactive play script for Belote with simplified 4-line interface.
"""

import os
import argparse
from src.stages.player.network import CNNBeloteNetwork
from src.stages.player.agent import PPOBeloteAgent
from src.stages.player.human import Human
from src.stages.player.simulation import simulate, create_env

def parse_args():
    parser = argparse.ArgumentParser(description="Play Belote against trained AI agents")
    parser.add_argument("--model-path", type=str, default="./models/belote_agent_final.pt", help="Model path for the agent")
    parser.add_argument("--type", type=str, default='play', choices=['play', 'observe', 'record'], help="Type of run: play, observe, record")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to play")
    return parser.parse_args()

def load_ai_agent(model_path):
    network = CNNBeloteNetwork()
    agent = PPOBeloteAgent(network=network)
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    
    return agent

def play_to_record():
    env = create_env()

    agents = []
    for i in range(0, 4):
        agent = Human()
        agent.reset(env, player=i)
        agents.append(agent)

    win, lose, history = simulate(env, agents, record=False, verbose=[0,1,2,3])

    history.save("path-to-save")

    print(f"You {"win :)" if win >= lose else "lose :()"} | points: {win} / {lose}")  

def play_to_observe(args):
    n = 0
    total_wins = 0
    agents = []
    for _ in range(0, 4):
        agent = load_ai_agent(args.model_path)
        agents.append(agent)

    while n < args.episodes:
        env = create_env()

        win, lose, _ = simulate(env, agents, record=False, verbose=None)

        n = n + 1
        total_wins += (win > lose) * 1
    
    print(f"Total played {n} | Wins: {total_wins} / {n - total_wins}")  

def play_against_agents():
    env = create_env()
    
    # Create a separate agent for each AI player
    agents = []
    for i in range(0, 4):
        agent = Human() if i == 0 else load_ai_agent(args.model_path)
        agents.append(agent)

    win, lose, _ = simulate(env, agents, record=False, verbose=[0])

    print(f"You {"win :)" if win >= lose else "lose :()"} | points: {win} / {lose}")  


if __name__ == "__main__":
    args = parse_args()

    if args.type == 'play':
        play_against_agents()
    if args.type == 'observe':
        play_to_observe(args)
    if args.type == 'record':
        play_to_record()