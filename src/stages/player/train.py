import os
import argparse
import numpy as np
import time
import torch
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
from simulation import play

def parse_args():
    root = os.environ.get("PROJECT_ROOT")
    save_path = os.path.join(root, 'models')
    os.makedirs(save_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a Belote agent using improved PPO")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for PPO updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--update-interval", type=int, default=10, help="Update agent every N episodes")
    parser.add_argument("--save-interval", type=int, default=100, help="Save model every N episodes")
    parser.add_argument("--eval-interval", type=int, default=200, help="Evaluate model every N episodes")
    parser.add_argument("--display", action="store_true", help="Display game progress")
    parser.add_argument("--save-path", type=str, default=save_path, help="Path to save models")
    parser.add_argument("--load-path", type=str, default=None, help="Path to load existing model")
    return parser.parse_args()

def train_agent(args):
    # Set random seed
    seed = int(time.time()) % 1000    
    
    # Initialize the neural network and agent
    network = CNNBeloteNetwork()
    agent = PPOBeloteAgent(
        network=network,
        training=True  # Set training to True for training mode
    )
    
    # Load existing model if specified
    if args.load_path:
        if agent.load(args.load_path):
            print(f"Loaded model from {args.load_path}")
        else:
            print(f"Could not load model from {args.load_path}, starting fresh")

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
        
        # Initialize agent with environment
        agent.init(env, env_index=0)
        
        # Use the play function from simulation.py to play the game
        team0_score, team1_score, _ = play(env, agents, history=None, display=args.display)
        
        # Update wins count if team 0 (with our agent) won
        if team0_score > team1_score:
            total_wins += 1
        
        # Update rewards for the agent's recent actions
        final_reward = (team0_score - team1_score) / 100.0  # Normalize the reward
        if len(agent.rewards) > 0:
            # Apply reward to recent actions with decay
            agent.updated_rewards(final_reward, last_n=min(8, len(agent.rewards)))
        
        # Perform learning every update_interval episodes
        if episode % args.update_interval == 0 and episode > 0 and len(agent.actions) > 0:
            # Learn from experiences
            agent.learn(batch_size=args.batch_size)
            
            # Reset batch after learning
            agent.reset_batch()

            print(f"Episode {episode}, Rounds Won: {total_wins}/{episode+1} "
                  f"({100*total_wins/(episode+1):.1f}%), "
                  f"Score: {team0_score} vs {team1_score}")
        
        # Save model at regular intervals
        if episode % args.save_interval == 0 and episode > 0:
            path = os.path.join(args.save_path, f"belote_agent_ep{episode}.pt")
            agent.save(path)
            print(f"Model saved to {path}")
        
        # Evaluation at regular intervals
        if episode % args.eval_interval == 0 and episode > 0:
            print(f"\n=== Evaluation at Episode {episode} ===")
            evaluate_agent(agent, random_agents, num_games=50)
            print("=" * 40)
    
    # Final learning if there are remaining experiences
    if len(agent.actions) > 0:
        agent.learn(batch_size=args.batch_size)
    
    # Save the final model
    final_path = os.path.join(args.save_path, "belote_agent_final.pt")
    agent.save(final_path)
    print(f"Training completed. Final model saved to {final_path}")

def evaluate_agent(agent, random_agents, num_games=50):
    """Evaluate the agent against random players"""
    agent.training = False  # Set to evaluation mode
    wins = 0
    total_score_diff = 0
    
    for game in range(num_games):
        # Setup evaluation environment
        trump = Trump()
        trump.set_random_trump()
        
        deck = Deck()
        deck.reset()
        deck.deal_cards(8)
        
        next_player = game % 4
        env = BeloteEnv(trump, deck, next_player=next_player)
        
        # Setup agents
        agents = [agent] + random_agents
        agent.init(env, env_index=0)
        
        # Play the game
        team0_score, team1_score, _ = play(env, agents, history=None, display=False)
        
        if team0_score > team1_score:
            wins += 1
        
        total_score_diff += (team0_score - team1_score)
    
    win_rate = wins / num_games
    avg_score_diff = total_score_diff / num_games
    
    print(f"Evaluation Results:")
    print(f"  Win Rate: {win_rate:.3f} ({wins}/{num_games})")
    print(f"  Average Score Difference: {avg_score_diff:.2f}")
    
    agent.training = True  # Set back to training mode
    return win_rate, avg_score_diff

if __name__ == "__main__":
    args = parse_args()
    train_agent(args)