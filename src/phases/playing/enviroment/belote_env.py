"""
BeloteEnv class for the Belote game environment.

3 main stages:
    1) Trick - When a cycle of 4 card plays ends (each player has played one card)
    2) Round - When the current play ends (no cards left in the deck)
    3) Game - When the total game ends (until reaching the point limit) - NOT IMPLEMENTED YET
"""
from typing import Tuple, List
from core.card import Card
from core.deck import Deck
from core.trump import Trump
from constants import Suits
try:
    from core.table import Table
except ImportError:
    pass # Table might be missing
from src.tricks.actions import Action, ActionPlayCard

class BeloteEnv:        
    def __init__(self, trump, deck, next_player=0):
        # States
        self.deck = deck
        self.trump = trump
        self.table = Table()
        self.trick = 0

        # Game state
        self.next_player = next_player
        self.total_scores = [0, 0] # i % 2
        self.trick_scores = [0, 0] # i % 2

        self.reset_trick()
    
    def reset_game(self, trump: Trump = None):
        self.total_scores = [0, 0]
        self.reset_trick()

    def reset_trick(self):
        self.trick = 9 - len(self.deck[self.next_player])
        self.trick_scores = [0, 0] # i % 2
        self.table.clear()

    def choose_action(self, action: Action) -> Tuple[int, bool, bool]:
        """Execute an action and return (next_player, trick_ended, round_ended)"""
        if isinstance(action, ActionPlayCard):
            return self._on_play_card(action.card)
        
        # Other action types
        # ...
        return self.next_player, False, False
    
    def valid_actions(self) -> List[Action]:
        """Get list of valid actions the current player can take"""

        valid_actions = []

        # Get valid play cards
        valid_play_cards = self._valid_play_cards()
        if len(valid_play_cards) > 0:
            valid_actions.extend(ActionPlayCard(card) for card in valid_play_cards)

        # Other action types
        # ...

        return valid_actions 

    def current_state(self) -> Trump, Hand, Table, Probabilities:
        """Get the current state of the game"""
        trump = self.trump
        hand = self.deck[self.next_player]
        table = self.table
        probabilities = self._calculate_probabilities()

        return trump, hand, table, probabilities

    def _valid_play_cards(self) -> List[Card]:
        """Get list of valid cards the current player can play"""
        # Check if the current player has any cards
        if not self.deck[self.next_player]:
            return []

        # When the player is the first to play, they can play any card
        if len(self.table) == 0:
            return self.deck[self.next_player]
        
        # Get lead suit (suit of the first card played in this trick)
        table_lead_suit = self.table[0].suit
        
        # Get highest trump card on the table
        table_winner_card, _ = self.table.winner_card(self.trump)
        
        # Get player's trump cards higher than the table's highest trump card
        player_winner_cards = [card for card in self.deck[self.next_player] 
                              if card.higher_than(table_winner_card, self.trump)]

        # Check if player has cards of the lead suit
        player_table_lead_suit_cards = [card for card in self.deck[self.next_player] 
                                       if card.suit == table_lead_suit]

        # When player has cards of the lead suit
        if player_table_lead_suit_cards:
            # When leading suit is trump
            if table_lead_suit == self.trump.suit:
                # If player has trump cards that are higher than the table's highest trump card
                if player_winner_cards:
                    return player_winner_cards
                    
            # Play any card of the lead suit
            return player_table_lead_suit_cards

        # Player has no cards of the lead suit
        # and player has winner cards 
        if player_winner_cards:
            return player_winner_cards
                
        # Player has no trump cards, they can play any card
        return self.deck[self.next_player]



    def _on_play_card(self, card: Card) -> Tuple[int, bool, bool]:
        """Play a card and return game state"""
        # Remove the card from the player's hand
        self.deck.remove_card(self.next_player, card)
        
        # Add the card to the table
        self.table.add(card)

        # Check if trick is complete
        if not self.table.is_full():
            # Update the current player for the next turn
            self.next_player = (self.next_player + 1) % 4            
            return self.next_player, False, False

        # Trick ended
        self._on_trick_end()
        
        # Check if round ended
        round_ended = self.deck.no_cards_left()

        return self.next_player, True, round_ended

    def _on_trick_end(self):
        """Handle end of trick - determine winner and update scores"""
        # When table is full, determine winner
        _, winner_position = self.table.winner_card(self.trump)

        # First Player - next_player is not updated yet that's why we need to add 1
        first_player = (self.next_player + 1) % 4

        # Update the next player to the winner of the trick
        self.next_player = (first_player + winner_position) % 4

        # Calculate trick points
        trick_points = self._trick_points()

        # Update the scores for the winning team
        self.trick_scores[self.next_player % 2] = trick_points
        self.total_scores[self.next_player % 2] += trick_points

    def _trick_points(self) -> int:
        """Calculate points for the current trick"""
        bonus = 10 if self.deck.no_cards_left() else 0
        return self.table.total_points(self.trump) + bonus

    def display_state(self):
        """Display the current game state"""
        if self.trump.mode == Trump.MODE_NO_TRUMP:
            trump_suit = "No"
        elif self.trump.mode == Trump.MODE_REGULAR:
            trump_suit = Suits[self.trump.suit]
        else:
            trump_suit = "Unknown"

        self.display_hash()
        print()
        print(f"Trick: {self.trick} | Next: {self.next_player + 1} | Trump: {trump_suit}")
        print()

    def display_hands(self, player: int = 0):
        """Display a player's hand"""
        player_cards = self.deck[player]
        print(f"Hands: {' '.join(str(card) for card in player_cards)}")

    def display_table(self):
        """Display cards on the table"""
        table_cards = [str(card) for card in self.table.cards]
        
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

