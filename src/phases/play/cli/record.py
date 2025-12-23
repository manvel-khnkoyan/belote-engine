import random
import sys
import os
import argparse
import torch

# Add the project root to the python path
sys.path.append(os.getcwd())

from src.phases.play.cli.utils.main import transform_canonical
from src.models.trump import Trump, TrumpMode
from src.utility.deck import Deck
from src.utility.cards import Cards
from src.phases.play.core.simulator import Simulator
from src.phases.play.core.rules import Rules
from src.phases.play.helper_agents.human import HumanAgent
from src.phases.play.ppo.agent import PpoAgent
from src.phases.play.ppo.network import PPONetwork

def main():
    parser = argparse.ArgumentParser(description="Record Belote Game")
    parser.add_argument("--model", type=str, help="Path to the trained model file (optional). If provided, plays against AI.")
    parser.add_argument("--output", type=str, default="records/record-001.pkl", help="Path to save the record file")
    args = parser.parse_args()

    # Randomly choose trump for demonstration
    trump = Trump(TrumpMode.Regular, random.randint(0,3))

    # Hands
    hands = [Cards.sort(hand, trump) for hand in Deck.create()]
    
    # Initialize agents
    if args.model:
        if not os.path.exists(args.model):
            print(f"Error: {args.model} not found.")
            return
        
        print(f"Loading model from {args.model}...")
        try:
            network = PPONetwork()
            network.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
            # Player 0 is Human, others are PPO
            agents = [HumanAgent()] + [PpoAgent(network) for _ in range(3)]
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        # All Human
        agents = [HumanAgent() for _ in range(4)]

    # Initialize Simulator
    rules = Rules()
    simulator = Simulator(rules, agents, display=True)

    # Transform to canonical form
    trump, hands = transform_canonical(trump, hands)

    print("")
    print("Starting Belote Game Recording...")
    print(f"Trump is {trump}")
    print("You will play for all 4 players.")
    
    # Run simulation
    # Assuming player 0 starts
    result = simulator.simulate(hands, trump, next_player=random.randint(0,3))
    
    print("\nGame Over!")
    print(f"Total Scores: {simulator.scores}")

    # Save the result
    save_path = args.output
    result.save(save_path)
    print(f"Game recorded to {save_path}")

if __name__ == "__main__":
    main()
