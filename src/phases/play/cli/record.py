import random
import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())

from src.phases.play.cli.utils.main import transform_canonical
from src.models.trump import Trump, TrumpMode
from src.utility.deck import Deck
from src.utility.cards import Cards
from src.phases.play.core.simulator import Simulator
from src.phases.play.core.rules import Rules
from src.phases.play.helper_agents.human import HumanAgent

def main():    
    # Randomly choose trump for demonstration
    trump = Trump(TrumpMode.Regular, random.randint(0,3))

    # Hands
    hands = [Cards.sort(hand, trump) for hand in Deck.create()]
    
    # Initialize agents - All Human
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
    save_path = "records/record-001.pkl"
    result.save(save_path)
    print(f"Game recorded to {save_path}")

if __name__ == "__main__":
    main()
