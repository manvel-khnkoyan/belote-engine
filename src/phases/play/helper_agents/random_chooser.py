import random
from sre_parse import State
from typing import Any, Dict, List, Tuple
from src.phases.play.core.agent import Agent
from src.phases.play.core.actions import Action

class RandomChooserAgent(Agent):
    def choose_action(self, _: State, actions: List[Action]) -> Tuple[Action, Dict[str, Any] | None]:
        return random.choice(actions), None
