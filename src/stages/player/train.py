import os
import argparse
import numpy as np
import time
import torch
from src.stages.player.env import BeloteEnv
from src.stages.player.network import BeloteNetwork
from src.stages.player.ppo.belote_agent import PPOBeloteAgent
from src.stages.player.ppo.memory import PPOMemory
from src.stages.player.helper_agents.randomer import Randomer
from src.states.trump import Trump
from src.deck import Deck
from simulation import play

def parse_args():
    root = os.environ.get("PROJECT_ROOT")
    save_path = os.path.join(root, 'models')
    os.makedirs(save_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a Belote agent using improved PPO")
    parser.add_argument("--episodes", type=int, default=100*64, help="Number of episodes to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--save-path", type=str, default=save_path, help="Path to save models")
    parser.add_argument("--load-path", type=str, default=None, help="Path to load existing model")
    return parser.parse_args()

def get_random_trump():
    """Get a random trump suit for the game"""
    trump = Trump()
    trump.set_random_trump()
    return trump

def get_random_deck():
    """Get a random deck of cards for the game"""
    deck = Deck()
    deck.reset()
    deck.deal_cards(8)
    return deck

def train_agent(args):
    # Set random seed
    seed = int(time.time()) % 1000    
    
    # Initialize the neural network and agent
    agent = PPOBeloteAgent(BeloteNetwork(), PPOMemory())
    # Load existing model if specified
    if args.load_path:
        agent.load(args.load_path)

    # Create random agents for other players
    random_agents = [Randomer() for _ in range(3)]
    
    # Total wins counter
    total_wins = 0
    
    print(f"Starting to collect experiences for {args.episodes} episodes...")

    # Training loop
    for _ in range(0, args.episodes):
        # Setup a new game environment
        seed = seed + 1

        # Play each game 4 consecutive times with the same deck
        deck = get_random_deck()
        trump = get_random_trump()
        next_player = int(np.random.default_rng(seed).integers(0, 4))

        # Setup the environment
        env = BeloteEnv(trump, deck, next_player=next_player)
        
        # Setup all agents - the PPO agent at position 0, and random agents at other positions
        agents = [agent] + random_agents
        
        # Initialize agent with environment
        agent.init(env, env_index=0)
        
        # Use the play function from simulation.py to play the game
        team0_score, team1_score = play(env, agents)
        
        # Update wins count if team 0 (with our agent) won
        total_wins += 1 if team0_score > team1_score else 0

        # Update agent's memory with the score difference
        reward = (team0_score - team1_score) / (team0_score + team1_score)
        agent.memory.updated_last_rewards(reward, last_n=1)


    print(f"Collected experiences for {args.episodes} episodes. "
          f"Total wins: {total_wins} ({100*total_wins/args.episodes:.1f}%)")
    
    print("Starting to learn from collected experiences..." )
    # Learn from experiences
    
    for _ in range(round(args.episodes / args.batch_size)):
        agent.learn(batch_size=args.batch_size)
        
    
    # Save the final model
    final_path = os.path.join(args.save_path, "belote_agent_final.pt")
    agent.save(final_path)

    print(f"Training completed. Final model saved to {final_path}")

if __name__ == "__main__":
    args = parse_args()
    train_agent(args)