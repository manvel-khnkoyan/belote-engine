"""
Training script for the Belote AI agent using PPO algorithm.
"""

import os
import argparse
import numpy as np
import time
from src.stages.player.env import BeloteEnv
from src.stages.player.network import CNNBeloteNetwork
from src.stages.player.agent import PPOBeloteAgent
from src.states.table import Table
from src.states.probability import Probability
from src.states.trump import Trump
from src.card import Card
from src.deck import Deck

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Belote agent using PPO")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--save-interval", type=int, default=100, help="Save model every N episodes")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate model every N episodes") 
    parser.add_argument("--output-dir", type=str, default="./models", help="Directory to save models")
    return parser.parse_args()

def setup_game(seed=None):
    trump = Trump()
    trump.set_random_trump()
    
    deck = Deck()
    deck.reset()
    deck.deal_cards(8)
    
    # Create environment with random initial player
    rng = np.random.default_rng(seed)
    next_player = int(rng.integers(0, 4))

    env = BeloteEnv(trump, deck, next_player=next_player)
    
    return env

def train_agent(args):
    # Set random seed
    seed = int(time.time()) % 1000

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the neural network and agent
    network = CNNBeloteNetwork()
    agent = PPOBeloteAgent(
        network=network,
        lr=args.lr,
        gamma=args.gamma
    )
    
    # Training loop
    for episode in range(1, args.episodes + 1):
        # Setup a new game environment
        seed = seed + 1
        env = setup_game(seed)
        agent.reset_memory(env, player=0)
        
        total_win = 0
        
        # Play until the game is over
        while True:
            current_player = env.next_player
            
            # If it's the agent's turn (player 0)
            if current_player == 0:
                # Choose an action using the agent
                card = agent.choose_action(env, memorize=True)
            else:
                # Choose random valid action for other players
                valid_cards = env.valid_cards()
                
                # Check if valid_cards is empty, which should not happen in a valid game state
                if len(valid_cards) == 0:
                    print(f"Warning: No valid cards for player {current_player}. Resetting game.")
                    # Skip this episode and try again
                    break
                
                # Set seed for this specific choice to ensure reproducibility
                seed = seed + 1
                rng = np.random.default_rng(seed)
                card = rng.choice(valid_cards)
            
            # Take the action in the environment
            _, trick_ended, round_ended = env.step(card)

            # Update each player's memory
            agent.update_memory(current_player, card)

            # calculate wins
            if trick_ended:
                total_win += 1 if env.trick_scores[0] > env.trick_scores[1] else 0

            if round_ended:
                total_reward = env.round_scores[0] - env.round_scores[1]
                agent.memory.update_last_reward(total_reward * 2)

                break

            if trick_ended:
                trick_reward = env.trick_scores[0] - env.trick_scores[1]
                agent.memory.update_last_reward(trick_reward)

                env.reset_trick()    
                    
        # Print progress and perform learning every 10 episodes
        if episode % 10 == 0 and len(agent.memory) > 0:
            # Learn from experiences with current trump for proper value estimation
            agent.learn(batch_size=min(args.batch_size, len(agent.memory)))
            
            # Only clear memory after learning to make use of all experiences
            agent.memory.clear()

            print(f"Episode {episode}, Round Won: {total_win} "
                  f"Score: {env.round_scores[0]} vs {env.round_scores[1]}")
        
        # Save model at regular intervals
        if episode % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"belote_agent_ep{episode}.pt")
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    # Save the final model
    final_path = os.path.join(args.output_dir, "belote_agent_final.pt")
    agent.save(final_path)
    print(f"Training completed. Final model saved to {final_path}")

if __name__ == "__main__":
    args = parse_args()
    train_agent(args)