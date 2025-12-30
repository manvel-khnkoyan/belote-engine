import os
import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple
from src.models.card import Card
from src.utility.deck import Deck
from src.utility.canonical import Canonical
from src.models.trump import Trump, TrumpMode
from src.phases.play.core.simulator import Simulator
from src.phases.play.core.rules import Rules
from src.phases.play.core.record import Record
from src.phases.play.core.agent import Agent
from src.phases.play.core.actions import ActionPlayCard, ActionPass, ActionShowSet, ActionAnnounceBelote
from src.phases.play.core.result import Result
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
        self.agent = PpoAgent(self.network, rng=self.rng)
        
        # Rules
        self.rules = Rules()

        # Predefined agent
        self.soft_player = SoftPlayerAgent()
        self.agent_player = PpoAgent(self.network, rng=self.rng)
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
            records, scores = self.play_games()
            
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
                    print(f"  Reward vs Loss: Avg Reward={scores:.2f}, Total Loss={metrics['total_loss']:.4f}")
            
            # Save model for next phase
            model_path = os.path.join(self.model_dir, "model.pt")
            self.save_model(model_path)

            # Reload agent with latest model
            network = PPONetwork().to(self.device)
            network.load_state_dict(torch.load(model_path, map_location=self.device))
            network.eval()
            
            # Update reference to the current network so next save works correctly
            self.network = network

            # Update BOTH the main agent and the opponent agent with the latest model
            self.agent = PpoAgent(network, rng=self.rng)
            self.agent_player = PpoAgent(network, rng=self.rng)
            
            phase_rewards.append(scores)

    def play_games(self) -> Tuple[List[Record], float]:
        # Play multiple games and collect experience for training.
        records: List[Record] = []
        total_scores = []
        
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
            
            # Compute GAE for card play
            self._compute_play_card_gae(result.records)

            # Collect records for PPO agent (player 0) and all records
            agent_records = [r for r in result.records if r.player == 0]
            
            # Append to overall records
            records.extend(agent_records)
            
            # Track rewards
            total_scores.append(result.scores[0] - result.scores[1])

        return records, np.mean(total_scores) if total_scores else 0.0

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
        print(f"\n{'='*30}{title}{'='*30}")
        
    def _get_random_trump(self) -> Trump:
        """Generate a random trump configuration."""
        number = self.rng.integers(0, 6) # 0-3: Regular, 4: NoTrump, 5: AllTrump

        if number < 4:
            return Trump(TrumpMode.Regular, number)
        if number == 4:
            return Trump(TrumpMode.NoTrump, None)
        if number == 5:
            return Trump(TrumpMode.AllTrump, None)

    def _compute_play_card_gae(self, records: List[Record], gamma: float = 0.99, lam: float = 0.95):
        play_records = [r for r in records if isinstance(r.action, ActionPlayCard)]
        if not play_records:
            return
        
        rewards = [0] * 8
        for r in play_records:
            if r.player in (1, 3):
                rewards[r.round] -= r.accrued_reward / 100.0
            if r.player == 2:
                rewards[r.round] += r.accrued_reward / (3 * 100.0) # reward but not that much
            if r.player == 0:
                rewards[r.round] += r.accrued_reward / (100.0)

        # extract only player 0 records
        player_play_records = [r for r in play_records if r.player == 0]
        
        if not player_play_records:
            return
        
        # extract values for player 0 - aligned with player_play_records
        values = [float(r.log['value']) for r in player_play_records]
        values.append(0)
        
        # Align rewards with player_play_records by their round
        aligned_rewards = [rewards[r.round] for r in player_play_records]
        
        gae = 0
        for i in reversed(range(len(player_play_records))):
            delta = aligned_rewards[i] + gamma * values[i + 1] - values[i]
            gae = delta + gamma * lam * gae
            
            player_play_records[i].log['advantage'] = gae
            player_play_records[i].log['return'] = gae + values[i]


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
