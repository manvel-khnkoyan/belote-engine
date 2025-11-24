import sys
import os

# Add the project root to the python path
# Assuming this file is in src/phases/play/cli/
# Root is ../../../../
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Also add src to path because some modules might import from 'const' directly
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
    
# Also add src/models to path because card.py imports 'trump' directly
models_path = os.path.join(src_path, "models")
if models_path not in sys.path:
    sys.path.append(models_path)

from src.models.deck import Deck
from src.models.trump import Trump, TrumpMode
from src.const import Suits
from src.phases.play.core.simulator import Simulator
from src.phases.play.core.rules import Rules
from src.phases.play.core.state import State
from src.phases.play.helper_agents.human import HumanAgent
from src.phases.play.helper_agents.random_chooser import RandomChooserAgent

def main():
    # Setup game elements
    deck = Deck()
    # Spades is 0
    trump = Trump(TrumpMode.Regular, 0) 
    
    # Create a dummy state for initialization
    dummy_state = State(0, [], trump)
    
    # Initialize agents
    # Player 0 is Human, 1, 2, 3 are Random
    agents = [
        HumanAgent(dummy_state),
        RandomChooserAgent(dummy_state),
        RandomChooserAgent(dummy_state),
        RandomChooserAgent(dummy_state)
    ]
    
    # Initialize Simulator
    rules = Rules()
    simulator = Simulator(rules)
    
    print("Starting Belote Game Simulation...")
    print(f"Trump is {trump}")
    print("You are Player 0.")
    
    # Run simulation
    # Assuming player 0 starts
    result = simulator.simulate(agents, deck, trump, next_player=0)
    
    print("\nGame Over!")
    print(f"Total Scores: {simulator.total_scores}")

if __name__ == "__main__":
    main()
