
import os
import argparse
import numpy as np
import time
from src.stages.player.env import BeloteEnv
from src.stages.player.network import CNNBeloteNetwork
from src.stages.player.agent import PPOBeloteAgent
from src.stages.player.randomer import Randomer
from src.states.table import Table
from src.states.probability import Probability
from src.states.trump import Trump
from src.card import Card
from src.deck import Deck
from src.stages.player.actions import ActionCardMove
from simulation import play  # Import the play function from simulation.py

def parse_args():
    root = os.environ.get("PROJECT_ROOT")
    save_path = os.path.join(root, 'models')
    os.makedirs(save_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a Belote agent using PPO")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--save-interval", type=int, default=100, help="Save model every N episodes")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate model every N episodes") 
    parser.add_argument("--display", action="store_true", help="Display game progress")
    parser.add_argument("--save-path", type=str, default=save_path, help="Display game progress")
    return parser.parse_args()

def train_agent(args):
    # Set random seed
    seed = int(time.time()) % 1000    
    
    # Initialize the neural network and agent
    network = CNNBeloteNetwork()
    agent = PPOBeloteAgent(
        network=network,
        lr=args.lr,
        gamma=args.gamma,
        memorize=True  # Set memorize to True for training
    )

    # Create random agents for other players
    random_agents = [Randomer() for _ in range(3)]
    
    total_wins = 0
    
    # Training loop
    for episode in range(0, args.episodes):
        # Setup a new game environment
        seed = seed + 1

        # Play each game 4 consecutive times with the same deck
        if episode % 4 == 0:
            trump = Trump()
            trump.set_random_trump()
            
            deck = Deck()
            deck.reset()
            deck.deal_cards(8)

            next_player = int(np.random.default_rng(seed).integers(0, 4))

        # Setup the environment
        env = BeloteEnv(trump, deck.copy(), next_player=next_player)
        
        # Setup all agents - the PPO agent at position 0, and random agents at other positions
        agents = [agent] + random_agents
        
        # Use the play function from simulation.py to play the game
        team0_score, team1_score, _ = play(env, agents, history=None, display=args.display)
        
        # Update wins count if team 0 (with our agent) won
        if team0_score > team1_score:
            total_wins += 1
        
        # Update final reward in agent's memory
        total_reward = team0_score - team1_score
        agent.memory.update_last_reward(total_reward * 2)  # Multiplied by 2 to maintain the same scale as before
        
        # Print progress and perform learning every 10 episodes
        if episode % 10 == 0 and len(agent.memory) > 0:
            # Learn from experiences
            agent.learn(batch_size=min(args.batch_size, len(agent.memory)))
            
            # Clear memory after learning
            agent.memory.clear()

            print(f"Episode {episode}, Rounds Won: {total_wins} "
                  f"Score: {team0_score} vs {team1_score}")
        
        # Save model at regular intervals
        if episode % args.save_interval == 0 and episode > 0:
            path = os.path.join(args.save_path, f"belote_agent_ep{episode}.pt")
            agent.save(path)
            print(path)
    
    # Save the final model
    final_path = os.path.join(args.save_path, "belote_agent_final.pt")
    agent.save(final_path)
    print(f"Training completed. Final model saved to {final_path}")

if __name__ == "__main__":
    args = parse_args()
    train_agent(args)