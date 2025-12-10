import random
from sre_parse import State
from typing import Any, Dict, List, Tuple

from src.phases.play.core.actions import Action, ActionPass
from src.phases.play.core.result import Result
from src.phases.play.ppo.agent import PpoAgent

class PpoTester(PpoAgent):
    # static variables for all instances
    cursor = 0
    total_moves = 0
    total_matches = 0

    def __init__(self, *args, result: Result):
        super().__init__(*args)
        PpoTester.result = result

    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Dict[str, Any] | None]:
        record = PpoTester.result.records[PpoTester.cursor]
        recorded_action = record.action
        PpoTester.cursor += 1

        agent_action, _ = super().choose_action(state, actions)

        if not isinstance(agent_action, ActionPass):
            if agent_action == recorded_action:
                PpoTester.total_matches += 1
            PpoTester.total_moves += 1   

        return recorded_action, None

        
