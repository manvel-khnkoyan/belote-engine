import copy
from .state import State
from .actions import Action

"""
Record represents a single played action in the game.
"""
class Record:
    def __init__(self, player: int, state: State, action: Action, instant_reward: int, accrued_reward: int):
        self.player = player
        self.state = copy.deepcopy(state)
        self.action = copy.deepcopy(action)
        # Rewards are 2: immediate and accrued
        self.instant_reward = instant_reward
        self.accrued_reward = accrued_reward

    def __repr__(self):
        try:
            return (
                f"Record(player={self.player}, action={repr(self.action)},"
                f"Reward={self.instant_reward}/{self.accrued_reward}, "
                f"{repr(self.state)}"
            )
        except Exception:
            return f"Record(player={self.player})"
