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
    parser.add_argument("--times", type=int, default=1, help="Number of times to play the game")
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

    # Initialize Rules
    rules = Rules()

    # Reset global statistics
    PpoTester.total_moves = 0
    PpoTester.total_matches = 0

    # Determine the starting player from the first record
    start_player = result.records[0].player

    print(f"Starting {args.times} comparison simulation(s)...")
    
    # Run simulations N times
    for game_num in range(args.times):
        print(f"Game {game_num + 1}/{args.times}")
        
        # Reset cursor for each game
        PpoTester.cursor = 0
        
        # Create new agents for this game
        agents = [PpoTester(network, result=result) for _ in range(4)]

        # Initialize Simulator
        simulator = Simulator(rules, agents, display=False)

        # Run simulation
        # We use the hands and trump from the recorded result
        simulator.simulate(result.hands, result.trump, next_player=start_player)
    
    # Print combined stats
    if PpoTester.total_moves > 0:
        accuracy = (PpoTester.total_matches / PpoTester.total_moves) * 100
        print("\nComparison Complete.")
        print(f"Total Games: {args.times}")
        print(f"Total Moves: {PpoTester.total_moves}")
        print(f"Matches: {PpoTester.total_matches}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo moves were simulated.")

if __name__ == "__main__":
    main()
