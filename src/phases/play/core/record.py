from .state import State
from .actions import Action

class Record:
    def __init__(self, player: int, round: int, state: State, action: Action, reward: int):
        self.player = player
        self.round = round
        self.state = state
        self.action = action
        self.reward = reward

