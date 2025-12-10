import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.phases.play.ppo.gym import Gym

def main():
    parser = argparse.ArgumentParser(description="Train Belote PPO Agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for PPO update")
    parser.add_argument("--model_path", type=str, default="models/belote_agent.pt", help="Path to save/load model")
    
    args = parser.parse_args()
    
    gym = Gym(model_path=args.model_path)
    gym.train(episodes=args.episodes, batch_size=args.batch_size, save_path=args.model_path)

if __name__ == "__main__":
    main()
