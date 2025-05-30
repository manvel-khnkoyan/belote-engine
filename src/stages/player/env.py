"""
BeloteEnv class for the Belote game environment.

3 main stages:
    1) Trick - When a cycle of 4 card plays ends (each player has played one card)
    2) Round - When the current play ends (no cards left in the deck)
    3) Game - When the total game ends (until reaching the point limit) - NOT IMPLEMENTED YET
"""
import numpy as np
from src.states.table import Table
from src.card import Card
from src.card import SUIT_SYMBOLS
from src.stages.player.actions import Action, ActionCardMove

class BeloteEnv:        
    def __init__(self, trump, deck, next_player=0):
        # States
        self.deck = deck
        self.trump = trump
        self.table = Table()

        # Game state
        self.next_player = next_player
        self.round_scores = [0, 0] # i % 2
        self.trick_scores = [0, 0] # i % 2

        self.reset_trick()

    def reset_trick(self):
        self.trick_scores = [0, 0] # i % 2
        self.table.clear()

    def valid_cards(self) -> list:
        # Check if the current player has any cards
        if not self.deck[self.next_player]:
            return []  # Player has no cards to play

        # When the player is the first to play, they can play any card
        if len(self.table) == 0:
            return self.deck[self.next_player]
        
        # Get lead suit (suit of the first card played in this trick)
        table_lead_suit = self.table[0].suit
        
        # Get highest trump card on the table
        table_winner_card, _ = self.table.winner_card(self.trump)
        
        # Check if player has cards of the lead suit
        player_table_lead_suit_cards = [card for card in self.deck[self.next_player] if card.suit == table_lead_suit]
        
        # Get current player's trump cards
        player_trump_cards = [card for card in self.deck[self.next_player] if card.is_trump(self.trump)]

        ### When player has cards of the lead suit
        if len(player_table_lead_suit_cards) > 0:
            # When leading suit is trump
            if self.table[0].is_trump(self.trump):
                # Get player's trump cards higher than the table's highest trump card
                player_winner_cards = [card for card in player_trump_cards if card.higher_than(table_winner_card, self.trump)]

                # If player has trump cards higher than current highest, they must play those
                if len(player_winner_cards) > 0:
                    return player_winner_cards
                    
            # Play any card of the lead suit
            return player_table_lead_suit_cards

        ### Player has no cards of the lead suit

        # If player has trump cards, they must play a trump
        if len(player_trump_cards) > 0:
            return player_trump_cards
                
        # Player has no trump cards, they can play any card
        return self.deck[self.next_player]
    
    def step(self, action: Action):
        if isinstance(action, ActionCardMove):
            # Play a card
            return self._play_card(action.card)
        
        # Other action types
        # ...

    def _play_card(self, card: Card):
        # we can check if the card is from valid cards from the next_player, 
        # but we assume the player will play a valid card
        # to simplify the code and reduce unnecessary calculations
        
        # Remove the card from the player's hand
        self.deck.remove_card(self.next_player, card)
        
        # Add the card to the table
        self.table.add(card)

        # No observation update needed here, it will be done when the table is full
        if not self.table.is_full():
            # Update the current player for the next turn
            self.next_player = (self.next_player + 1) % 4            
            return self.next_player, False, False

        # if trick ended
        if (trick_ended := self.table.is_full()):
            self._on_trick_end()

        # if round ended
        round_ended = self.deck.no_cards_left()

        return self.next_player, trick_ended, round_ended

    def _on_trick_end(self):
        # When table is full, determine winner
        _, winner_position = self.table.winner_card(self.trump)

        # First Player | next_player is not updated yet that's why we need to add 1
        first_player = (self.next_player + 1) % 4

        # Update the next player to the winner of the trick
        self.next_player = (first_player + winner_position) % 4

        # Trick scores
        trick_points = self._trick_points()

        # Update the scores for the winning team
        self.trick_scores[self.next_player % 2] = trick_points
        self.round_scores[self.next_player % 2] += trick_points

    def _trick_points(self):
        # last trick has a bonus
        bonus = 10 if self.deck.no_cards_left() else 0
        # Calculate the points for the current trick
        return self.table.total_points(self.trump) + bonus

    def display_state(self):
        # Find the trump suit
        if np.any(self.trump.values == 1):
            trump_suit_index = np.argmax(self.trump.values)
            trump_suit = SUIT_SYMBOLS[trump_suit_index]
        else:
            trump_suit = "No"
            
        # Get player's cards
        game_round = 9 - min([len(hand) for hand in self.deck.hands])

        self.display_line()
        print()
        print(f"Round: {game_round} | Next: {self.next_player} | Trump: {trump_suit}")
        print()

    def display_hands(self, player=0):            
        # Get player's cards
        player_cards = self.deck[player]

        self.display_line()
        print(f"Hands: {' '.join([f"{str(player_cards[i])}" for i in range(len(player_cards))])}")
        print(f"Index: {''.join([f" {i + 1}  " for i in range(len(player_cards))])}")
        print()

    def display_table(self, end=None):
        table_cards = []
        for card in self.table.cards:
            if card is not None:
                # Simply show all cards without indicating who played them
                table_cards.append(str(card))
        
        # Fill with placeholders if needed
        while len(table_cards) < 4:
            table_cards.append(" . ")
        
        print(f"Table: {" ".join(table_cards)}", end=end)

    def display_available_cards(self, player, end=None):
        available_cards = self.valid_cards()
        available_cards_indices = [str(i + 1) for i, card in enumerate(self.deck[player]) if card in available_cards]

        print(f"[{','.join(available_cards_indices)}]", end=end)

    def display_summary(self):
        round_gain = self.round_scores[0]
        round_loss = self.round_scores[1]

        win_or_lost = '[0]' if round_gain > round_loss else '[1]'

        self.display_line()
        print()
        print(f"Total: Winner is {win_or_lost}, Total ({round_gain}, {round_loss})")
        print()
        self.display_line()

    
    def display_line(self):
        print("------------------------------------------")

