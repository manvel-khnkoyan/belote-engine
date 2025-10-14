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
from trump import Trump
from src.deck import Deck
from src.stages.player.simulator import simulate
from src.stages.player.network_monitor import NetworkMonitor

def parse_args():
    root = os.environ.get("PROJECT_ROOT")
    save_path = os.path.join(root, 'models')
    os.makedirs(save_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a Belote agent using cyclic self-play PPO")
    parser.add_argument("--episodes", type=int, default=10*640, help="Number of episodes per training session")
    parser.add_argument("--sessions", type=int, default=5, help="Number of training sessions/cycles")
    parser.add_argument("--batch-size", type=int, default=640, help="Batch size for PPO updates")
    parser.add_argument("--save-path", type=str, default=save_path, help="Path to save models")
    parser.add_argument("--load-path", type=str, default=None, help="Path to load existing model to start from")
    return parser.parse_args()

def get_random_deck():
    """Get a random deck of cards for the game"""
    deck = Deck()
    deck.reset()
    deck.deal_cards(8)
    return deck

def create_agents(model_path=None):
    agents = []
    for _ in range(4):
        agent = PPOBeloteAgent(BeloteNetwork(), PPOMemory())
        if model_path is not None:
            agent.load(model_path)
        agents.append(agent)
    
    # The first agent is our main training agent, others are opponents
    return agents

 # Callback for when a trick ends to update rewards
def create_on_trick_end(env, agents):
    # Calculate the point differential for the trick
    # trick_scores[0] is team 0 (our agent's team)
    # trick_scores[1] is team 1 (opponent team)
    trick_reward = env.trick_scores[0] - env.trick_scores[1]

    # Normalize the reward to a smaller, more stable range (e.g., -1 to 1)
    # Max possible points in a trick is around 30-50. Let's use a conservative normalizer.
    normalized_reward = trick_reward / 50.0

    # The agent played one of the last 4 cards. We give the reward to the last action.
    agents[0].memory.updated_last_rewards(normalized_reward, last_n=1)

def train_session(args, agents):
    """Optimized training session with better monitoring"""

    # Set memory for the main agent
    agents[0].memory = PPOMemory()
    seed = int(time.time()) % 1000
    wins = 0
    rate = 0

    print(f"* Collecting experiences for {args.episodes} episodes...")

    # Episode collection loop
    for _ in range(args.episodes):
        deck = get_random_deck()
        trump = Trump.random()
        next_player = int(np.random.default_rng(seed).integers(0, 4))
        env = BeloteEnv(trump, deck, next_player=next_player)

        # Create the on_trick_end callback
        on_trick_end = create_on_trick_end(env, agents)

        # simulate
        simulate(env, agents, on_trick_end=on_trick_end, display=False)

        # Get scores from environment after simulation
        team0_score, team1_score = env.total_scores[0], env.total_scores[1]
        total_score = team0_score + team1_score

        wins += 1 if team0_score > team1_score else 0
        rate = team0_score / total_score

        # Reduce signal amplification to lower KL divergence
        reward = (team0_score - team1_score) / total_score
        reward = reward * 1.5  # Reduced from 2.0 for more stable learning
        agents[0].memory.updated_last_rewards(reward, last_n=2)

    print(f"* Experiences completed: {wins}/{args.episodes} rate ({rate:.1f}%)")
    print()
    
    print("[ Learning started ]")

    # Create batches of indices from 0 to indices_size
    seed += 1
    memory = agents[0].memory
    indices_size = len(memory)
    rng = np.random.default_rng(seed=seed)
    indices_shuffled = rng.permutation(indices_size)
    for i in range(0, indices_size, args.batch_size):
        indices = indices_shuffled[i:i + args.batch_size]
        # learn by sequential sampling
        agents[0].learn(memory.sample(indices))

        # Display diagnostics
        agents[0].ppo.display_diagnostics()

    # Print progress - Remove the problematic entropy/KL calculation
    # The original code was trying to access 'entropy' and 'kl_divergence' from memory.actions,
    # but memory.actions contains integers (action IDs), not dictionaries
    print("[ Learning completed ]")


def train(args):
    """Main training function with cyclic self-play"""
    print("Starting cyclic self-play training:")
    print(f" . Sessions: {args.sessions}")
    print(f" . Episodes per session: {args.episodes}")
    print(f" . Batch size: {args.batch_size}")
    print(f" . Save path: {args.save_path}")
    
    # Initialize model path
    current_model_path = args.load_path if args.load_path else None

    # Initialize agents
    agents = []
    
    for session in range(1, args.sessions + 1):
        print(f"\n====== SESSION {session} ======")

        # Create agents for this session
        agents = create_agents(
            model_path=current_model_path,
        )
        
        # Train for this session
        train_session(args, agents)
        
        # Save the model after this session
        session_model_path = os.path.join(args.save_path, f"belote_agent_session_{session}.pt")
        agents[0].save(session_model_path)
        
        # Update current model path for next session
        current_model_path = session_model_path
    
    # Save the final model
    final_path = os.path.join(args.save_path, "belote_agent.pt")
    agents[0].save(final_path)

    print("\n====== TRAINING COMPLETED ======")
    print(f"Total sessions: {args.sessions}")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)