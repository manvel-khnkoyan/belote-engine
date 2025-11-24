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
from .actions import ActionPlayCard
from .record import Record

class Simulator:
    def __init__(self, rules: Rules):
        self.rules = rules
        self.total_scores = [0, 0]
        self.trick_scores = [0, 0]
        self.trick_number = 1

    def simulate(self, agents: List[Agent], deck: Deck, trump: Trump, next_player: int) -> Result:
        hands = deck.deal()
        table = []
        records = []
        
        # 32 cards total
        cards_played = 0
        self.trick_number = 1
        self.total_scores = [0, 0]
        
        while cards_played < 32:
            current_player = next_player
            
            # Display state before action
            self.display_state(trump, next_player)
            self.display_hands(hands, current_player)
            self.display_table(table)
            
            # Create state for current player
            state = State(current_player, list(hands[current_player]), trump)
            state.table = list(table) # Copy table
            
            # Get valid actions
            actions = self.rules.actions(state)
            
            # Agent chooses action
            agent = agents[current_player]
            agent.state = state
            action = agent.choose_action(actions)
            
            # Apply action
            if isinstance(action, ActionPlayCard):
                card = action.card
                
                # Remove from hand
                if card in hands[current_player]:
                    hands[current_player].remove(card)
                
                # Add to table
                table.append(card)
                
                # Record the move
                records.append(Record(current_player, cards_played // 4, state, action, 0))
                
                # Advance player
                next_player = (next_player + 1) % 4
                cards_played += 1
                
                # Check trick end
                if len(table) == 4:
                    self.display_table(table) # Show full table before clearing
                    
                    # Determine winner
                    # The player who started the trick is 'next_player' (since we incremented 4 times mod 4)
                    trick_starter = next_player
                    
                    table_collection = Collection(table)
                    winner_card, winner_idx = table_collection.winner(trump)
                    
                    # winner_idx is 0..3 relative to table order
                    winner_player = (trick_starter + winner_idx) % 4
                    
                    # Calculate points
                    points = self._trick_points(table_collection, trump, cards_played == 32)
                    self.trick_scores[winner_player % 2] = points
                    self.total_scores[winner_player % 2] += points
                    
                    print(f"Trick Winner: Player {winner_player} (+{points} points)")
                    self.display_line()
                    
                    # Set winner as next player
                    next_player = winner_player
                    
                    # Clear table
                    table = []
                    self.trick_number += 1
            else:
                # Handle other actions
                pass
        
        self.display_summary()
        return Result(deck, trump, records)

    def _trick_points(self, table: Collection, trump: Trump, last_trick: bool) -> int:
        """Calculate points for the current trick"""
        points = sum(card.value(trump) for card in table)
        bonus = 10 if last_trick else 0
        return points + bonus

    def display_state(self, trump: Trump, next_player: int):
        """Display the current game state"""
        if trump.mode == TrumpMode.NoTrump:
            trump_suit = "No"
        elif trump.mode == TrumpMode.Regular:
            trump_suit = Suits[trump.suit]
        elif trump.mode == TrumpMode.AllTrump:
            trump_suit = "All"
        else:
            trump_suit = "Unknown"

        self.display_hash()
        print()
        print(f"Trick: {self.trick_number} | Next: Player {next_player} | Trump: {trump_suit}")
        print()

    def display_hands(self, hands: List[List[Card]], player: int):
        """Display a player's hand"""
        player_cards = hands[player]
        print(f"Player {player} Hand: {' '.join(str(card) for card in player_cards)}")

    def display_table(self, table: List[Card]):
        """Display cards on the table"""
        table_cards = [str(card) for card in table]
        
        # Fill with placeholders if needed
        while len(table_cards) < 4:
            table_cards.append(" . ")
        
        print(f"Table: {' '.join(table_cards)}")

    def display_summary(self):
        """Display final round summary"""
        round_gain = self.total_scores[0]
        round_loss = self.total_scores[1]
        win_or_lost = '[0]' if round_gain > round_loss else '[1]'

        self.display_hash()
        print()
        print(f"Total: Winner is {win_or_lost}, Total ({round_gain}, {round_loss})")
        print()
        self.display_hash()
    
    def display_line(self):
        """Display a separator line"""
        print("------------------------------------------")
        
    def display_hash(self):
        """Display a hash separator"""
        print("##########################################")
