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

    parser = argparse.ArgumentParser(description="Train a Belote agent using cyclic self-play PPO")
    parser.add_argument("--episodes", type=int, default=100*64, help="Number of episodes per training session")
    parser.add_argument("--sessions", type=int, default=5, help="Number of training sessions/cycles")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--save-path", type=str, default=save_path, help="Path to save models")
    parser.add_argument("--load-path", type=str, default=None, help="Path to load existing model to start from")
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

def create_agents(model_path=None):
    agents = []
    for _ in range(4):
        agent = PPOBeloteAgent(BeloteNetwork(), PPOMemory())
        if model_path is not None:
            agent.load(model_path)
        agents.append(agent)
    
    # The first agent is our main training agent, others are opponents
    return agents[0], agents[1:]

def train_session(main_agent, opponent_agents, episodes, batch_size, session_num):
    """Train for one session with the given agents"""
    print(f"\n=== SESSION {session_num} ===")
    print(f"Training agent type: {type(main_agent).__name__}")
    print(f"Opponent agent types: {[type(agent).__name__ for agent in opponent_agents]}")
    
    # Reset the main agent's memory for fresh experience collection
    main_agent.memory = PPOMemory()
    
    # Set random seed
    seed = int(time.time()) % 1000 + session_num * 1000
    
    # Total wins counter for this session
    session_wins = 0
    
    print(f"Collecting experiences for {episodes} episodes...")

    # Training loop for this session
    for episode in range(episodes):
        # Setup a new game environment
        seed = seed + 1

        # Play game with random deck and trump
        deck = get_random_deck()
        trump = get_random_trump()
        next_player = int(np.random.default_rng(seed).integers(0, 4))

        # Setup the environment
        env = BeloteEnv(trump, deck, next_player=next_player)
        
        # Setup all agents - the main PPO agent at position 0, opponents at other positions
        all_agents = [main_agent] + opponent_agents
        
        # Initialize all PPO agents with environment
        for i, agent in enumerate(all_agents):
            if hasattr(agent, 'init'):  # PPO agents have init method
                agent.init(env, env_index=i)
        
        # Use the play function from simulation.py to play the game
        team0_score, team1_score = play(env, all_agents)
        
        # Update wins count if team 0 (with our main agent) won
        session_wins += 1 if team0_score > team1_score else 0

        # Update main agent's memory with the score difference
        reward = (team0_score - team1_score) / (team0_score + team1_score)
        main_agent.memory.updated_last_rewards(reward, last_n=1)

        # Print progress occasionally
        if (episode + 1) % (episodes // 10) == 0:
            win_rate = 100 * session_wins / (episode + 1)
            print(f"  Episode {episode + 1}/{episodes}, Win rate: {win_rate:.1f}%")

    session_win_rate = 100 * session_wins / episodes
    print(f"Session {session_num} completed: {session_wins}/{episodes} wins ({session_win_rate:.1f}%)")
    
    print("Learning from collected experiences...")
    # Learn from experiences
    for _ in range(round(episodes / batch_size)):
        main_agent.learn(batch_size=batch_size)
    
    return session_win_rate

def train(args):
    """Main training function with cyclic self-play"""
    print("Starting cyclic self-play training:")
    print(f"  Sessions: {args.sessions}")
    print(f"  Episodes per session: {args.episodes}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save path: {args.save_path}")
    
    # Track progress across sessions
    session_win_rates = []
    
    # Initialize model path
    current_model_path = args.load_path if args.load_path else None
    
    for session in range(1, args.sessions + 1):
        # Create agents for this session
        main_agent, opponent_agents = create_agents(
            model_path=current_model_path,
        )
        
        # Train for this session
        win_rate = train_session(
            main_agent, 
            opponent_agents, 
            args.episodes, 
            args.batch_size, 
            session
        )
        session_win_rates.append(win_rate)
        
        # Save the model after this session
        session_model_path = os.path.join(args.save_path, f"belote_agent_session_{session}.pt")
        main_agent.save(session_model_path)
        print(f"Session {session} model saved to: {session_model_path}")
        
        # Update current model path for next session
        current_model_path = session_model_path
        
        # Print progress summary
        print(f"\nProgress Summary after {session} sessions:")
        for i, wr in enumerate(session_win_rates, 1):
            print(f"  Session {i}: {wr:.1f}% win rate")
    
    # Save the final model
    final_path = os.path.join(args.save_path, "belote_agent_final.pt")
    main_agent.save(final_path)
    
    print("\n=== TRAINING COMPLETED ===")
    print(f"Total sessions: {args.sessions}")
    print(f"Final model saved to: {final_path}")
    print("Win rate progression:")
    for i, wr in enumerate(session_win_rates, 1):
        print(f"  Session {i}: {wr:.1f}%")

if __name__ == "__main__":
    args = parse_args()
    train(args)