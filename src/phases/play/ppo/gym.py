import os
import torch
import numpy as np
from typing import List
from src.models.card import Card
from src.utility.deck import Deck
from src.utility.canonical import Canonical
from src.models.trump import Trump, TrumpMode
from src.phases.play.core.simulator import Simulator
from src.phases.play.core.rules import Rules
from src.phases.play.core.record import Record
from src.phases.play.ppo.agent import PpoAgent
from src.phases.play.ppo.network import PPONetwork
from src.phases.play.helper_agents.random_chooser import RandomChooserAgent

class Gym:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Network and Agent
        self.network = PPONetwork().to(self.device)
        self.agent = PpoAgent(self.network)
        
        if model_path and os.path.exists(model_path):
            try:
                self.load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model from {model_path}: {e}")
            
        # Initialize Opponents
        self.opponents = [RandomChooserAgent() for _ in range(3)]
        
        # Rules
        self.rules = Rules()

    def train(self, episodes: int = 1000, batch_size: int = 64, save_path: str = "model.pt"):
        print(f"Starting training for {episodes} episodes...")
        
        buffer: List[Record] = []
        total_rewards = []
        
        for episode in range(1, episodes + 1):
            # Setup Game
            hands = Deck.create(shuffle=True)
            
            # Random Trump
            trump_suit = np.random.randint(0, 4)
            trump = Trump(TrumpMode.Regular, trump_suit)

            # Convert Canonical
            canonical = Canonical.create_transform_map(hands[0])
            trump.suit = canonical[trump.suit] if trump.suit is not None else None
            hands = [[Card(canonical[card.suit], card.rank) for card in hand] for hand in hands]
            
            # Assign Agents (PPO is player 0)
            agents = [self.agent] + self.opponents
            
            # Simulate
            simulator = Simulator(self.rules, agents, display=False)
            # Assuming player 0 starts for simplicity, or random
            start_player = np.random.randint(0, 4)
            
            result = simulator.simulate(hands, trump, start_player)
            
            # Filter records for PPO agent (player 0)
            agent_records = [r for r in result.records if r.player == 0]
            
            # Compute GAE and update logs
            self._compute_gae(agent_records)
            
            buffer.extend(agent_records)
            
            # Track performance
            episode_reward = sum(r.instant_reward for r in agent_records) + (agent_records[-1].accrued_reward if agent_records else 0)
            total_rewards.append(episode_reward)
            
            # Train if buffer is full
            if len(buffer) >= batch_size:
                self.agent.learn(buffer)
                buffer = [] # Clear buffer after update (On-Policy)
                
            if episode % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                print(f"Episode {episode}/{episodes} | Avg Reward: {avg_reward:.2f}")
                
            if episode % 100 == 0 and save_path:
                self.save_model(save_path)
                
        if save_path:
            self.save_model(save_path)
            print(f"Training complete. Model saved to {save_path}")

    def _compute_gae(self, records: List[Record], gamma=0.99, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE) and Returns.
        Updates record.log with 'advantage' and 'return'.
        """
        if not records:
            return
            
        values = [r.log['value'] for r in records]
        rewards = [r.instant_reward for r in records]
        
        # Add final accrued reward to the last reward
        rewards[-1] += records[-1].accrued_reward
        
        # Append 0 for value of terminal state
        values.append(0.0)
        
        gae = 0
        for i in reversed(range(len(records))):
            delta = rewards[i] + gamma * values[i+1] - values[i]
            gae = delta + gamma * lam * gae
            
            records[i].log['advantage'] = gae
            records[i].log['return'] = gae + values[i]

    def save_model(self, path: str):
        self.network.save(path)

    def load_model(self, path: str):
        self.network.load(path)
