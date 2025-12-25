import os
import torch
import numpy as np
from typing import List, Tuple
from src.models.card import Card
from src.utility.deck import Deck
from src.utility.canonical import Canonical
from src.models.trump import Trump, TrumpMode
from src.phases.play.core.simulator import Simulator
from src.phases.play.core.rules import Rules
from src.phases.play.core.record import Record
from src.phases.play.core.state import State
from src.phases.play.ppo.agent import PpoAgent
from src.phases.play.ppo.network import PPONetwork
from src.phases.play.helper_agents.aggresive_player import AggressivePlayerAgent
from src.phases.play.helper_agents.soft_player import SoftPlayerAgent
from src.phases.play.helper_agents.random_chooser import RandomChooserAgent

class Gym:
    def __init__(self, model_dir: str = "models", seed: int = 42):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize RNG with seed for reproducibility
        self.rng = np.random.default_rng(seed)
        
        # Initialize Network and Agent
        self.network = PPONetwork().to(self.device)
        self.agent = PpoAgent(self.network)
        
        # Rules
        self.rules = Rules()
        
        print(f"Gym initialized on device: {self.device}")

    def train_phases(
        self, 
        num_phases: int = 5,
        games_per_phase: int = 100,
        opponent_types: List[str] = None
    ):
        """
        Train PPO agent through multiple phases.
        
        Args:
            num_phases: Number of training phases
            games_per_phase: Number of games per phase
            opponent_types: List of opponent types ['random', 'aggressive', 'soft']
                           If None, defaults to random 3 opponents
        """
        if opponent_types is None:
            opponent_types = ['random', 'random', 'random']
        
        print("\nStarting Phase-Based Training")
        print(f"Total Phases: {num_phases}")
        print(f"Games per Phase: {games_per_phase}")
        print(f"Opponent Types: {opponent_types}\n")
        
        phase_rewards = []
        
        for phase in range(1, num_phases + 1):
            print(f"\n{'='*60}")
            print(f"PHASE {phase}/{num_phases}")
            print(f"{'='*60}")
            
            # Load model from previous phase (if available)
            if phase > 1:
                model_path = os.path.join(self.model_dir, f"model-phase-{phase-1}.pt")
                if os.path.exists(model_path):
                    self.load_model(model_path)
                    print(f"✓ Loaded model from phase {phase-1}")
            
            # Play games and collect experience
            records, phase_reward = self.play_games(
                num_games=games_per_phase,
                opponent_types=opponent_types
            )
            
            # Train on collected experience
            if records:
                print(f"\nTraining on {len(records)} experiences...")
                self.agent.learn(records)
                print("✓ Training complete")
            
            # Save model for next phase
            model_path = os.path.join(self.model_dir, "model.pt")
            self.save_model(model_path)
            print(f"✓ Model saved to {model_path}")
            
            phase_rewards.append(phase_reward)
            print(f"Phase Reward: {phase_reward:.2f}")
        
        # Summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        for i, reward in enumerate(phase_rewards, 1):
            print(f"Phase {i}: {reward:.2f}")
        print(f"Average: {np.mean(phase_rewards):.2f}")

    def play_games(
        self, 
        num_games: int = 100,
        opponent_types: List[str] = None
    ) -> Tuple[List[Record], float]:
        """
        Play multiple games and collect experience.
        
        Args:
            num_games: Number of games to play
            opponent_types: List of opponent types for the 3 opponents
            
        Returns:
            records: List of Record objects for training
            avg_reward: Average reward across games
        """
        if opponent_types is None:
            opponent_types = ['random', 'random', 'random']
        
        all_records: List[Record] = []
        total_rewards = []
        
        for game in range(1, num_games + 1):
            # Setup game
            hands = Deck.create(shuffle=True)
            
            # Random trump
            trump_suit = self.rng.integers(0, 4)
            trump = Trump(TrumpMode.Regular, trump_suit)
            
            # Canonicalize (transform suits based on hand strength)
            can_map, _ = Canonical.create_transform_map(hands[0], trump)
            trump.suit = can_map[trump.suit] if trump.suit is not None else None
            hands = [
                [Card(can_map[card.suit], card.rank) for card in hand] 
                for hand in hands
            ]
            
            # Create opponents
            opponents = [self._create_opponent(opp_type) for opp_type in opponent_types]
            
            # Assign agents (PPO is player 0)
            agents = [self.agent] + opponents
            
            # Simulate game
            simulator = Simulator(self.rules, agents, display=False)
            start_player = self.rng.integers(0, 4)
            
            result = simulator.simulate(hands, trump, start_player)
            
            # Collect records for PPO agent (player 0)
            agent_records = [r for r in result.records if r.player == 0]
            
            # Compute GAE
            self._compute_gae(agent_records)
            
            all_records.extend(agent_records)
            
            # Track reward
            game_reward = sum(r.instant_reward for r in agent_records) + (agent_records[-1].accrued_reward if agent_records else 0)
            total_rewards.append(game_reward)
            
            #if game % 10 == 0:
            #    avg_10 = np.mean(total_rewards[-10:])
            #    print(f"  Game {game}/{num_games} | Avg Reward (last 10): {avg_10:.2f}")
        
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        return all_records, avg_reward

    def _create_opponent(self, opponent_type: str):
        """
        Create an opponent agent of the specified type.
        
        Args:
            opponent_type: 'random', 'aggressive', or 'soft'
            
        Returns:
            Agent instance
        """
        opponent_type = opponent_type.lower()
        
        if opponent_type == 'random':
            return RandomChooserAgent()
        elif opponent_type == 'aggressive':
            return AggressivePlayerAgent()
        elif opponent_type == 'soft':
            return SoftPlayerAgent()
        else:
            print(f"Unknown opponent type '{opponent_type}', defaulting to random")
            return RandomChooserAgent()

    def _compute_gae(self, records: List[Record], gamma: float = 0.99, lam: float = 0.95):
        """
        Compute Generalized Advantage Estimation (GAE) and Returns.
        Updates record.log with 'advantage' and 'return'.
        """
        if not records:
            return
        
        # Extract values (ensure they are floats)
        values = [float(r.log['value']) for r in records]
        
        # Normalize rewards! (Crucial for stability)
        # Belote max score is ~162. Dividing by 100 keeps it in reasonable range.
        rewards = [r.instant_reward / 100.0 for r in records]
        
        # Add final accrued reward to the last reward
        if records:
            rewards[-1] += (records[-1].accrued_reward / 100.0)
        
        # Append 0 for value of terminal state
        values.append(0.0)
        
        gae = 0
        for i in reversed(range(len(records))):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            gae = delta + gamma * lam * gae
            
            records[i].log['advantage'] = gae
            records[i].log['return'] = gae + values[i]

    def save_model(self, path: str):
        """Save network state to disk."""
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str):
        """Load network state from disk."""
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.network.to(self.device)
        else:
            print(f"Warning: Model file not found at {path}")
