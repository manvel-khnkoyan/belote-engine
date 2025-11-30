import random
from src.models.deck import Deck
from src.models.trump import Trump, TrumpMode
from src.phases.play.core.simulator import Simulator
from src.phases.play.core.rules import Rules
from src.phases.play.core.state import State
from src.phases.play.helper_agents.human import HumanAgent
from src.phases.play.helper_agents.random_chooser import RandomChooserAgent

def main():
    # Setup game elements
    deck = Deck()
    # Randomly choose trump for demonstration
    trump = Trump(TrumpMode.Regular, random.randint(0,3)) 
    
    # Initialize agents
    agents = [HumanAgent() if i == 0 else RandomChooserAgent() for i in range(4)]

    # Initialize Simulator
    rules = Rules()
    simulator = Simulator(rules, agents, display=True)
    
    print("Starting Belote Game Simulation...")
    print(f"Trump is {trump}")
    print("You are Player 0.")
    
    # Run simulation
    # Assuming player 0 starts
    result = simulator.simulate(deck, trump, next_player=random.randint(0,3))
    
    print("\nGame Over!")
    print(f"Total Scores: {simulator.scores}")

    print("Print Result as JSON:")
    print(result)

if __name__ == "__main__":
    main()
