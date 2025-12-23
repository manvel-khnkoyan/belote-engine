import os
import torch
import argparse

from src.phases.play.core.result import Result
from src.phases.play.core.rules import Rules
from src.phases.play.ppo.network import PPONetwork
from src.phases.play.core.simulator import Simulator
from src.phases.play.helper_agents.ppo_tester import PpoTester

def main():
    parser = argparse.ArgumentParser(description="Test Belote PPO Agent against recorded game")
    parser.add_argument("--model", type=str, default="models/model.pt", help="Path to the trained model file")
    parser.add_argument("--record", type=str, default="records/record-001.pkl", help="Path to the record file")
    args = parser.parse_args()

    # Load the recorded game
    load_path = args.record
    if not os.path.exists(load_path):
        print(f"Error: {load_path} not found. Please run record.py first.")
        return

    print(f"Loading recorded game from {load_path}...")
    result = Result.load(load_path)

    if not result.records:
        print("No records found in the loaded game.")
        return

    # Initialize PPO Agent
    print("Loading PPO Agent...")
    network = PPONetwork()
    model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run training first.")
        return

    network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Loaded model from {model_path}")
    
    # Initialize PpoTester
    # We use separate instances for each player, but they share the cursor via the class
    PpoTester.cursor = 0
    agents = [PpoTester(network, result=result) for _ in range(4)]

    # Initialize Rules
    rules = Rules()

    # Initialize Simulator
    # We can enable display to see the game progress
    simulator = Simulator(rules, agents, display=False)

    print("Starting comparison simulation...")
    
    # Determine the starting player from the first record
    start_player = result.records[0].player
    
    # Run simulation
    # We use the hands and trump from the recorded result
    simulator.simulate(result.hands, result.trump, next_player=start_player)
    
    # Print stats
    if PpoTester.total_moves > 0:
        accuracy = (PpoTester.total_matches / PpoTester.total_moves) * 100
        print(f"\nComparison Complete.")
        print(f"Total Moves: {PpoTester.total_moves}")
        print(f"Matches: {PpoTester.total_matches}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo moves were simulated.")

if __name__ == "__main__":
    main()
