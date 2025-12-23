import random
import torch
import argparse
import os
from src.phases.play.cli.utils.main import transform_canonical
from src.phases.play.ppo.agent import PpoAgent
from src.suits import Suits
from src.models.card import Card
from src.models.trump import Trump, TrumpMode
from src.utility.deck import Deck
from src.utility.cards import Cards
from src.utility.canonical import Canonical
from src.phases.play.core.simulator import Simulator
from src.phases.play.core.rules import Rules
from src.phases.play.helper_agents.human import HumanAgent
from src.phases.play.helper_agents.random_chooser import RandomChooserAgent
from src.phases.play.ppo.network import PPONetwork

def main():
    parser = argparse.ArgumentParser(description="Play Belote against PPO Agent")
    parser.add_argument("--model", type=str, default="models/model.pt", help="Path to the trained model file")
    args = parser.parse_args()

    # Randomly choose trump for demonstration
    trump = Trump(TrumpMode.Regular, random.randint(0,3))

    # Hands
    hands = [Cards.sort(hand, trump) for hand in Deck.create()]
    
    # Initialize agents
    network = PPONetwork()
    
    network.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    print(f"Loaded model from {args.model}")

    agents = [HumanAgent()] + [PpoAgent(network) for _ in range(3)]

    # Initialize Simulator
    rules = Rules()
    simulator = Simulator(rules, agents, display=True)

    # Transform to canonical form
    trump, hands = transform_canonical(trump, hands)

    print("")
    print("Starting Belote Game Simulation...")
    print(f"Trump is {trump}")
    print("You are Player 0.")
    
    # Run simulation
    # Assuming player 0 starts
    result = simulator.simulate(hands, trump, next_player=random.randint(0,3))
    
    print("\nGame Over!")
    print(f"Total Scores: {simulator.scores}")

    print(result)

if __name__ == "__main__":
    main()
