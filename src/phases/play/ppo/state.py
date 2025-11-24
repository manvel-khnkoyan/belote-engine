from ..core.state import State
from ....models.probability import Probability

class PpoState(State):
    def __init__(self, cards, trump, probability=None):
        super().__init__(cards, trump)
        self.probability: Probability = probability if probability is not None else Probability()

    def observe(self, player: int, action):
        super().observe(player, action)

        # Update probability based on action
        self.probability.update(player, action)


