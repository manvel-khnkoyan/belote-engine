import random
from typing import List
from src.phases.play.core.agent import Agent
from src.phases.play.core.actions import Action

class RandomChooserAgent(Agent):
    def choose_action(self, actions: List[Action]) -> Action:
        return random.choice(actions)
