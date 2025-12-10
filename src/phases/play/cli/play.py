import random
import torch
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
    # Randomly choose trump for demonstration
    trump = Trump(TrumpMode.Regular, random.randint(0,3))

    # Hands
    hands = [Cards.sort(hand, trump) for hand in Deck.create()]

    # Create canonical mapping based on player 0's hand
    canonical_map = Canonical.create_transform_map(hands[0])
    
    # Transform trump and hands using canonical mapping
    Suits.transform(canonical_map)
    trump = Trump(trump.mode, canonical_map[trump.suit]) if trump.mode != TrumpMode.NoTrump else trump
    hands = [[Card(canonical_map[card.suit], card.rank) for card in hand] for hand in hands]
    
    # Initialize agents
    # agents = [HumanAgent() if i == 0 else RandomChooserAgent() for i in range(4)]
    network = PPONetwork() 
    network.load_state_dict(torch.load("models/belote_agent.pt", map_location=torch.device('cpu')))

    agents = [HumanAgent()] + [PpoAgent(network) for _ in range(3)]

    # Initialize Simulator
    rules = Rules()
    simulator = Simulator(rules, agents, display=True)
    
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
