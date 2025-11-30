import random
from sre_parse import State
from typing import List
from src.phases.play.core.agent import Agent
from src.phases.play.core.actions import Action

class RandomChooserAgent(Agent):
    def choose_action(self, _: State, actions: List[Action]) -> Action:
        return random.choice(actions)
