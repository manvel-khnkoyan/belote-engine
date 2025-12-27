import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.phases.play.ppo.gym import Gym

def main():
    parser = argparse.ArgumentParser(description="Train Belote PPO Agent")
    parser.add_argument("--phases", type=int, default=5, help="Number of training phases, default is 5")
    parser.add_argument("--games-per-phase", type=int, default=100, help="Number of games per phase, default is 100")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save/load models, default is 'models'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility, default is 42")
    parser.add_argument("--opponents", type=str, default="random,aggressive,soft,agent", 
                        help="Comma-separated opponent types: random, aggressive, soft, agent")
    
    args = parser.parse_args()
    
    # Parse opponent types
    opponent_types = [opp.strip() for opp in args.opponents.split(",")]
    if len(opponent_types) < 3:
        print("Error: Must specify at least 3 opponent types")
        sys.exit(1)
    
    gym = Gym(model_dir=args.model_dir, seed=args.seed)
    gym.train_phases(
        num_phases=args.phases,
        games_per_phase=args.games_per_phase,
        opponent_types=opponent_types
    )

if __name__ == "__main__":
    main()
