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
from src.phases.play.core.agent import Agent
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

        # Predefined agent
        self.soft_player = SoftPlayerAgent()
        self.agent_player = PpoAgent(self.network)
        self.randomer_player = RandomChooserAgent()
        self.aggressive_player = AggressivePlayerAgent()
        
        print(f"Gym initialized on device: {self.device}")

    def train_phases(
        self, 
        opponent_types: List[str],
        num_phases: int,
        games_per_phase: int,
    ):
        self.opponent_types = opponent_types
        self.num_phases = num_phases
        self.games_per_phase = games_per_phase
        
        print("\nStarting Phase-Based Training")
        print(f"Total Phases: {self.num_phases}")
        print(f"Games per Phase: {self.games_per_phase}")
        print(f"Opponent Types: {self.opponent_types}\n")
        
        phase_rewards = []
        
        for phase in range(1, self.num_phases + 1):
            self._display_title(f"PHASE {phase}/{self.num_phases}")
            
            # Play games and collect experience
            records, phase_reward = self.play_games()
            
            # Train on collected experience
            if records:
                print(f"\nTraining on {len(records)} experiences...")
                metrics = self.agent.learn(records)
                print("✓ Training complete")
                if metrics:
                    print(f"  Loss: {metrics['total_loss']:.4f} "
                          f"(Policy: {metrics['policy_loss']:.4f}, "
                          f"Value: {metrics['value_loss']:.4f}, "
                          f"Entropy: {metrics['entropy']:.4f})")
            
            # Save model for next phase
            model_path = os.path.join(self.model_dir, "model.pt")
            self.save_model(model_path)

            # Reload agent with latest model
            network = PPONetwork().to(self.device)
            network.load_state_dict(torch.load(model_path, map_location=self.device))
            network.eval()
            self.agent_player = PpoAgent(network)
            
            phase_rewards.append(phase_reward)
            print(f"Phase Reward: {phase_reward:.2f}")
        
        # Summary
        self._display_title("TRAINING")
       
        print(f"Average: {np.mean(phase_rewards):.2f}")
        print(f"Best: {np.max(phase_rewards):.2f}")
        print(f"first: {phase_rewards[0]:.2f}")
        print(f"Last: {phase_rewards[-1]:.2f}")

    def play_games(self) -> Tuple[List[Record], float]:
        # Play multiple games and collect experience for training.
        all_records: List[Record] = []
        total_rewards = []
        
        for _ in range(1, self.games_per_phase + 1):
            # Setup game
            hands = Deck.create(shuffle=True)
            
            # Random trump
            trump = self._get_random_trump()
            
            # Canonicalize (transform suits based on hand strength)
            can_map, _ = Canonical.create_transform_map(hands[0], trump)
            trump.suit = can_map[trump.suit] if trump.suit is not None else None
            hands = [[Card(can_map[card.suit], card.rank) for card in hand] for hand in hands]
            
            # Create opponents
            # If more than 3 opponent types are provided, randomly sample 3
            opponents = self._select_opponents(self.opponent_types)
            
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
        
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        return all_records, avg_reward

    def _select_opponents(self, opponent_names: str) -> List[Agent]:
        assert len(opponent_names) >= 3, "At least 3 opponent types must be provided."

        players = {
            'random': self.randomer_player,
            'aggressive': self.aggressive_player,
            'soft': self.soft_player,
            'agent': self.agent_player
        }

        assert all(opp_type.lower() in players for opp_type in opponent_names), \
            f"Invalid opponent types in {opponent_names}. Valid types are: {list(players.keys())}"
        
        # Randomly select 3 opponents from the provided types
        selected_opponents = self.rng.choice(opponent_names, size=3, replace=True)

        return [players[opp_type.lower()] for opp_type in selected_opponents]

    def _display_title(self, title: str):
        print(f"\n{'='*60}{title}{'='*60}")
        
    def _get_random_trump(self) -> Trump:
        """Generate a random trump configuration."""
        number = self.rng.integers(0, 4) # 0-3: Regular, 4: NoTrump, 5: AllTrump

        if number < 4:
            return Trump(TrumpMode.Regular, number)
        if number == 4:
            return Trump(TrumpMode.NoTrump, None)
        if number == 5:
            return Trump(TrumpMode.AllTrump, None)

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
