from typing import List
from src.models.deck import Deck
from src.models.trump import Trump, TrumpMode
from src.models.collection import Collection
from src.models.card import Card
from src.const import Suits
from .rules import Rules
from .result import Result
from .agent import Agent
from .state import State
from .actions import ActionPlayCard, ActionPass
from .record import Record

class Simulator:
    def __init__(self, rules: Rules, agents: List[Agent], display: bool = False):
        self.rules = rules
        self.agents = agents
        self.display = display

    def simulate(self, deck: Deck, trump: Trump, next_player: int, states: List[State] = []) -> Result:
        # Initialize game state
        self.scores = [0, 0]
        self.trump = trump
        
        # Deal and sort hands
        hands = [Collection(hand).sort(trump) for hand in deck.deal()]

        # State
        self.states = states if states else [State(hands[i], trump) for i in range(4)]

        # Records
        records: List[Record] = []
        
        # 32 cards total
        cards_moves = 0

        # Player last played record (dictionary)
        player_last_card_played_record_index = {}

        # Display Line
        self._display_line()
        
        # Unitil All cards played
        while cards_moves < 32:
            
            # Start of a new trick
            current_player = next_player # Absolute player index 0..3 by agent order
            current_agent = self.agents[current_player]
            current_state = self.states[current_player]
            
            # Get valid actions
            actions = self.rules.actions(current_state)
            
            # Agent chooses action
            action = current_agent.choose_action(current_state, actions)

            # Recompute the current table snapshot after observers updated state
            current_table = Collection(current_state.table)

            #----------------------------#
            #---- Other actions here ----#
            #----------------------------#
            # .....

            #----------------------------#
            #----- CARD PLAY ACTION -----#
            #----------------------------#
            if isinstance(action, ActionPlayCard):
                # Update Table
                current_table.append(action.card)

                # Display Table after action
                self._display_table(current_table)

                # Update Card Moves
                cards_moves += 1

                # Winner card and its index on the table
                _, winner_table_idx = current_table.winner(trump)

                # Table points
                table_points = current_table.value(trump)

                # Winner player index (absolute). Compute start of trick from
                # the current player and number of cards on the table, then map
                # the winner table index to absolute player index.
                winner_player_index = (current_player - (len(current_table) - 1) + winner_table_idx) % 4

                # Instant reward for the player
                instant_reward = table_points if winner_player_index == current_player else 0
                
                # Record the move
                player_last_card_played_record_index[current_player] = len(records)

                # Append record
                records.append(Record(current_player, current_state, action, instant_reward, 0))

                # Continue to next player
                next_player = (current_player + 1) % 4
                
                # END OF TRICK PROCESSING
                if cards_moves % 4 == 0:
                    # Lazy reward update (accrued) for the last played cards
                    records[player_last_card_played_record_index[winner_player_index]].accrued_reward = table_points
                    
                    # Calculate points
                    self.scores[winner_player_index % 2] += table_points
                    
                    # Set winner as next player
                    next_player = winner_player_index

                    # Display Table after action
                    self._display_line()

            # Update all states 
            for i in range(4):
                self.states[i].observe(self._relative_player(i, current_player), action)

        self._display_summary()

        return Result(deck, trump, records)

    def _trick_points(self, table: Collection, trump: Trump, last_trick: bool) -> int:
        """Calculate points for the current trick"""
        points = sum(card.value(trump) for card in table)
        bonus = 10 if last_trick else 0
        return points + bonus
    
    def _relative_player(self, current_player: int, relative_to: int) -> int:
        """Get player index relative to another player"""
        return (current_player - relative_to) % 4

    def _display_table(self, current_table: Collection):
        if not self.display:
            return

        """Display cards on the table"""
        table_cards = [str(card) for card in current_table]
        
        # Fill with placeholders if needed
        while len(table_cards) < 4:
            table_cards.append(" . ")
        
        print(f"Table: {' '.join(table_cards)}")

    def _display_summary(self):
        if not self.display:
            return

        """Display final round summary"""
        round_gain = self.scores[0]
        round_loss = self.scores[1]
        win_or_lost = '[0]' if round_gain > round_loss else '[1]'

        self._display_line()
        print(f"Total: Winner is {win_or_lost}, Total ({round_gain}, {round_loss})")
        self._display_line()
        
    def _display_line(self):
        """Display a hash separator"""
        print("")
        print("-----------------------------------------")
        print("")
